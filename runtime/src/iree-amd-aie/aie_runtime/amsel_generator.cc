// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "amsel_generator.h"

#include <deque>
#include <limits>

#define DEBUG_TYPE "iree-aie-runtime-amsel-generator"

namespace mlir::iree_compiler::AMDAIE {

LogicalResult MSelGenerator::addConnection(
    const PhysPortAndID &src, const SmallVector<PhysPortAndID> &dsts) {
  SmallVector<Port> dstPorts =
      llvm::map_to_vector(dsts, [](const PhysPortAndID &physPortAndID) {
        return physPortAndID.physPort.port;
      });
  if (!connectionToMSel.contains(dstPorts)) {
    if (curMSel >= numMSels) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Trying to assign msel (" << std::to_string(curMSel)
                 << ") larger than number of msels available ("
                 << std::to_string(numMSels) << ") for " << src << "\n");
      return failure();
    }
    connectionToMSel[dstPorts] = curMSel++;
  }
  physPortToConnection[src] = dstPorts;
  srcPhysPortAndIDs.push_back(src);
  return success();
}

LogicalResult MSelGenerator::addConnections(
    const SmallVector<PhysPortAndID> &srcs,
    const SmallVector<SmallVector<PhysPortAndID>> &dsts) {
  for (auto &&[src, dstsElem] : llvm::zip(srcs, dsts)) {
    if (failed(addConnection(src, dstsElem))) return failure();
  }
  return success();
}

uint8_t MSelGenerator::getNumMSelsForConnections(
    const SmallVector<PhysPortAndID> &srcs,
    const SmallVector<SmallVector<PhysPortAndID>> &dsts) const {
  uint8_t numMSelsNeeded{0};
  for (const SmallVector<PhysPortAndID> &dstsElem : dsts) {
    assert(numMSelsNeeded < std::numeric_limits<uint8_t>::max());
    SmallVector<Port> dstPorts =
        llvm::map_to_vector(dstsElem, [](const PhysPortAndID &physPortAndID) {
          return physPortAndID.physPort.port;
        });
    if (!connectionToMSel.contains(dstPorts)) numMSelsNeeded++;
  }
  return numMSelsNeeded;
}

std::optional<uint8_t> MSelGenerator::getMSel(
    const PhysPortAndID &physPortAndID) const {
  if (!physPortToConnection.contains(physPortAndID)) return std::nullopt;
  SmallVector<Port> connectionKey = physPortToConnection.at(physPortAndID);
  if (!connectionToMSel.contains(connectionKey)) return std::nullopt;
  return connectionToMSel.at(connectionKey);
}

SmallVector<std::pair<PhysPortAndID, uint8_t>> MSelGenerator::getPhysPortMSels()
    const {
  SmallVector<std::pair<PhysPortAndID, uint8_t>> res;
  for (const PhysPortAndID &physPortAndID : srcPhysPortAndIDs) {
    std::optional<uint8_t> maybeMSel = getMSel(physPortAndID);
    assert(maybeMSel && "port is missing an msel assignemt");
    res.push_back(std::make_pair(physPortAndID, maybeMSel.value()));
  }
  return res;
}

void TileAMSelGenerator::addConnection(
    const PhysPortAndID &srcPhysPort,
    const SmallVector<PhysPortAndID> &dstPhysPorts) {
  srcToDstPorts[srcPhysPort].append(dstPhysPorts.begin(), dstPhysPorts.end());
  for (const PhysPortAndID &dstPhysPort : dstPhysPorts) {
    LLVM_DEBUG(llvm::dbgs() << "Add connection: " << srcPhysPort << " ->  "
                            << dstPhysPort << "\n");
    dstToSrcPorts[dstPhysPort].push_back(srcPhysPort);
    dstPortToPhysPortAndID[dstPhysPort.physPort.port].push_back(dstPhysPort);
    dstPorts.push_back(dstPhysPort.physPort.port);
  }
}

/// Find an `(arbiter, msel)` solution for all specified connections under the
/// following constraints and priorities:
/// 1. Prioritize arbiters over msels to avoid msel constraints and for better
/// performance.
/// 2. Constraint: all 'port and id's that share the same destination port
/// (channel), need to share the same arbiter.
/// 3. Optimization: try to reuse msels for connections that have the same
/// destination ports. For example, if two different sources (possibly using the
/// same physical port) route to the same physical destination ports (e.g.
/// {StrmSwPortType::SOUTH, 0}), then they can share the same msel value (and
/// arbiter).
///
/// This function solves the problem by combining 'port and id's into groups
/// that share some destination port. As all 'port and id' elements in a group
/// will need to share the arbiter, different groups will get a different
/// arbiter assigned if possible and msels are assigned withing a group.
/// However, when there are more groups than available arbiters, groups will be
/// merged/assigned the same arbiter if there are still msels available.
LogicalResult TileAMSelGenerator::solve() {
  // Clear all currently assigned `(arbiter, msel)` pairs.
  portsToAMSels.clear();
  // Keep track of a set of physical 'port and ids' that belong to the same
  // group.
  SmallVector<SmallVector<PhysPortAndID>> portGroups;
  DenseSet<PhysPortAndID> visitedPhysPorts;
  DenseSet<Port> visitedPorts;
  for (const Port &dstPort : dstPorts) {
    if (visitedPorts.contains(dstPort)) continue;
    SmallVector<PhysPortAndID> portGroup;
    std::deque<PhysPortAndID> q;
    q.insert(q.end(), dstPortToPhysPortAndID.at(dstPort).begin(),
             dstPortToPhysPortAndID.at(dstPort).end());
    while (q.size() > 0) {
      PhysPortAndID elem = q.front();
      q.pop_front();
      LLVM_DEBUG(llvm::dbgs() << "Queue pop elem: " << elem << "\n");
      if (visitedPhysPorts.contains(elem)) continue;
      visitedPhysPorts.insert(elem);
      if (srcToDstPorts.contains(elem)) {
        portGroup.push_back(elem);
        q.insert(q.end(), srcToDstPorts.at(elem).begin(),
                 srcToDstPorts.at(elem).end());
      } else {
        Port port = elem.physPort.port;
        if (visitedPorts.contains(port)) continue;
        visitedPorts.insert(port);
        q.insert(q.end(), dstPortToPhysPortAndID.at(port).begin(),
                 dstPortToPhysPortAndID.at(port).end());
        for (PhysPortAndID physPortAndID : dstPortToPhysPortAndID[port]) {
          q.insert(q.end(), dstToSrcPorts.at(physPortAndID).begin(),
                   dstToSrcPorts.at(physPortAndID).end());
        }
      }
    }
    portGroups.push_back(portGroup);
  }
  LLVM_DEBUG(llvm::dbgs() << "Number of groups created: " << portGroups.size()
                          << "\n");
  // Assign a new arbiter for each source group if available. Otherwise, try to
  // merge groups by utilizing msels.
  uint8_t arbiterIndex{0};
  // Map from an arbiter index to a corresponding msel generator utility.
  DenseMap<uint8_t, MSelGenerator> arbiterToMSelGenerator;
  for (const SmallVector<PhysPortAndID> &group : portGroups) {
    SmallVector<PhysPortAndID> srcs(group.begin(), group.end());
    SmallVector<SmallVector<PhysPortAndID>> dsts = llvm::map_to_vector(
        srcs, [&](const PhysPortAndID &src) -> SmallVector<PhysPortAndID> {
          return srcToDstPorts.at(src);
        });
    if (arbiterIndex < numArbiters) {
      MSelGenerator mselGenerator(numMSels);
      if (failed(mselGenerator.addConnections(srcs, dsts))) return failure();
      arbiterToMSelGenerator[arbiterIndex++] = std::move(mselGenerator);
    } else {
      bool foundMerge{false};
      for (uint8_t a = 0; a < arbiterIndex; a++) {
        MSelGenerator &mselGenerator = arbiterToMSelGenerator[a];
        uint8_t numMSelsNeeded =
            mselGenerator.getNumMSelsForConnections(srcs, dsts);
        if (mselGenerator.getCurMSel() + numMSelsNeeded < numMSels) {
          foundMerge = true;
          if (failed(mselGenerator.addConnections(srcs, dsts)))
            return failure();
          break;
        }
      }
      if (!foundMerge) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "All arbiters are used and could not merge remaining groups\n");
        return failure();
      }
    }
  }
  // Retrieve `(arbiter, msel)` pairs for all source ports.
  for (uint8_t arbiter = 0; arbiter < arbiterIndex; arbiter++) {
    const MSelGenerator &mselGenerator = arbiterToMSelGenerator.at(arbiter);
    for (auto &&[physPortAndID, msel] : mselGenerator.getPhysPortMSels()) {
      portsToAMSels[physPortAndID] = {arbiter, msel};
    }
  }
  return success();
}

std::optional<std::pair<uint8_t, uint8_t>> TileAMSelGenerator::getAMSel(
    const PhysPortAndID &port) {
  if (portsToAMSels.contains(port)) return portsToAMSels.at(port);
  LLVM_DEBUG(llvm::dbgs() << "Unknown PhysPortAndID: " << port);
  return std::nullopt;
}

void AMSelGenerator::initTileIfNotExists(TileLoc tileLoc, uint8_t numArbiters,
                                         uint8_t numMSels) {
  if (!tileToAMSelConfig.contains(tileLoc))
    tileToAMSelConfig[tileLoc] = TileAMSelGenerator(numArbiters, numMSels);
}

LogicalResult AMSelGenerator::addConnection(
    TileLoc tileLoc, const PhysPortAndID &srcPort,
    const SmallVector<PhysPortAndID> &dstPorts) {
  if (!tileToAMSelConfig.contains(tileLoc)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Can't add a connection on an unitialized tile: " << tileLoc);
    return failure();
  }
  tileToAMSelConfig[tileLoc].addConnection(srcPort, dstPorts);
  return success();
}

std::optional<std::pair<uint8_t, uint8_t>> AMSelGenerator::getAMSel(
    TileLoc tileLoc, const PhysPortAndID &port) {
  if (!tileToAMSelConfig.contains(tileLoc)) return std::nullopt;
  return tileToAMSelConfig[tileLoc].getAMSel(port);
}

LogicalResult AMSelGenerator::solve() {
  for (auto &&[tileLoc, tileAMSelGen] : tileToAMSelConfig) {
    LLVM_DEBUG(llvm::dbgs() << "Solve tile: " << tileLoc << "\n");
    if (failed(tileAMSelGen.solve())) return failure();
  }
  return success();
}

}  // namespace mlir::iree_compiler::AMDAIE
