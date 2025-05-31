// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_AMDAIE_AMSEL_GENERATOR_H_
#define IREE_COMPILER_AMDAIE_AMSEL_GENERATOR_H_

#include <numeric>
#include <utility>

#include "iree_aie_router.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::iree_compiler::AMDAIE {

/// Utility to assign msels to physical 'port and ID's based on the connection
/// they belong to. For the purposes of assigning msel values, a connection is
/// defined by its set of destination ports (stream switch type, channel). I.e.
/// different connections with the same destination ports can reuse msels.
class MSelGenerator {
 public:
  MSelGenerator() {}
  MSelGenerator(uint8_t numMSels) : numMSels(numMSels) {}

  /// Adds a connection and assigns an msel value. Returns failure if no msel
  /// could be assigned.
  LogicalResult addConnection(const PhysPortAndID &src,
                              const SmallVector<PhysPortAndID> &dsts);

  /// Adds connections and assigns an msel values. Returns failure if not all
  /// connections could get an msel value assigned.
  LogicalResult addConnections(
      const SmallVector<PhysPortAndID> &srcs,
      const SmallVector<SmallVector<PhysPortAndID>> &dsts);

  /// Return the number of msel values needed to add the specified connections.
  /// This can be used by users to ensure that this generator doesn't run out of
  /// available msel values.
  uint8_t getNumMSelsForConnections(
      const SmallVector<PhysPortAndID> &srcs,
      const SmallVector<SmallVector<PhysPortAndID>> &dsts) const;

  /// Return the current msel value to be used for the next new connection.
  uint8_t getCurMSel() const { return curMSel; }

  /// Return the assigned msel value for the provided physical port. Returns
  /// `std::nullopt` if the port doesn't have any assigned.
  std::optional<uint8_t> getMSel(const PhysPortAndID &physPortAndID) const;

  /// Return the assigned `(port, msel)` pairs inside this generator.
  SmallVector<std::pair<PhysPortAndID, uint8_t>> getPhysPortMSels() const;

 private:
  /// The number of available master/dest selects per switch.
  uint8_t numMSels{0};
  /// The current msel value to be used for the next new connection.
  uint8_t curMSel{0};
  /// Keep track of all source physical ports added to this generator for later
  /// retrieval of `(port, msel)` pairs.
  SmallVector<PhysPortAndID> srcPhysPortAndIDs;
  /// Map from connection keys to assigned msel values.
  DenseMap<SmallVector<Port>, uint8_t> connectionToMSel;
  /// Map from physical source ports and IDs to connection keys.
  DenseMap<PhysPortAndID, SmallVector<Port>> physPortToConnection;
};

/// Utility to generate arbiter - msel (master/dest select) pairs for every
/// connection add to this tile.
class TileAMSelGenerator {
 public:
  TileAMSelGenerator() {}
  TileAMSelGenerator(uint8_t numArbiters, uint8_t numMSels)
      : numArbiters(numArbiters), numMSels(numMSels) {}

  /// Add a connection from the specified source to the specified destinations.
  /// This does not assign any amsel pairs, but keeps track of the connections
  /// to be solved later.
  void addConnection(const PhysPortAndID &srcPhysPort,
                     const SmallVector<PhysPortAndID> &dstPhysPorts);

  /// Returns a `(arbiter, msel)` assignment for the provided port if one is
  /// found.
  std::optional<std::pair<uint8_t, uint8_t>> getAMSel(
      const PhysPortAndID &port);

  /// Tries to find a valid `(arbiter, msel)` assignment for all the added
  /// connections.
  LogicalResult solve();

 private:
  /// The number of available arbiters per switch.
  uint8_t numArbiters{0};
  /// The number of available master/dest selects per switch.
  uint8_t numMSels{0};
  /// Keep track of all the destination switch ports added.
  SmallVector<Port> dstPorts;
  /// Map from destination ports to the physical destination port-id pairs using
  /// them.
  DenseMap<Port, SmallVector<PhysPortAndID>> dstPortToPhysPortAndID;
  /// Map containing added connections from the source side.
  DenseMap<PhysPortAndID, SmallVector<PhysPortAndID>> srcToDstPorts;
  /// Map containing added connections from the destination side.
  DenseMap<PhysPortAndID, SmallVector<PhysPortAndID>> dstToSrcPorts;
  /// Map from source 'physical port and id' to the assigned `(arbiter, msel)`
  /// pair after solving for all connections.
  DenseMap<PhysPortAndID, std::pair<uint8_t, uint8_t>> portsToAMSels;
};

/// Utility to generate arbiter - msel (master/dest select) pairs for every
/// added connection. Use in the following way:
/// 1. Add all connections as sets of source and destination ports and IDs on a
/// specific tile.
/// 2. Solve the arbiter - msel selection problem by calling `solve()`. This
/// will return `failure` if no solution could be found, for example due to
/// running out of available arbiters and/or msels.
/// 3. Retrieve the `(arbiter, msel)` pairs for each connection by providing the
/// source port and ID of the connection.
class AMSelGenerator {
 public:
  AMSelGenerator() = default;

  /// Create a TileAMSelGenerator for the provided tile location with the
  /// specified number of arbiters and msels. Expected to be called for all
  /// tiles that will be used for connections.
  void initTileIfNotExists(TileLoc tileLoc, uint8_t numArbiters,
                           uint8_t numMSels);

  /// Add a connection from the specified source to the specified destinations
  /// on the provided tile location. This does not assign any amsel pairs, but
  /// keeps track of the connections to be solved later.
  LogicalResult addConnection(TileLoc tileLoc, const PhysPortAndID &srcPort,
                              const SmallVector<PhysPortAndID> &dstPorts);

  /// Returns a `(arbiter, msel)` assignment for the provided port if one is
  /// found.
  std::optional<std::pair<uint8_t, uint8_t>> getAMSel(
      TileLoc tileLoc, const PhysPortAndID &port);

  /// Find an arbiter - msel solution for all the added connections on all the
  /// tiles.
  LogicalResult solve();

 private:
  /// Keep track of a separate generator per tile location.
  DenseMap<TileLoc, TileAMSelGenerator> tileToAMSelConfig;
};

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_COMPILER_AMDAIE_AMSEL_GENERATOR_H_
