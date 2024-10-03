// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "pcidev.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <filesystem>
#include <fstream>

#include "amdxdna_accel.h"
#include "bo.h"
#include "pcidrv.h"
#include "shim_debug.h"

namespace {

std::string ioctl_cmd2name(unsigned long cmd) {
  switch (cmd) {
    case DRM_IOCTL_AMDXDNA_CREATE_HWCTX:
      return "DRM_IOCTL_AMDXDNA_CREATE_HWCTX";
    case DRM_IOCTL_AMDXDNA_DESTROY_HWCTX:
      return "DRM_IOCTL_AMDXDNA_DESTROY_HWCTX";
    case DRM_IOCTL_AMDXDNA_CONFIG_HWCTX:
      return "DRM_IOCTL_AMDXDNA_CONFIG_HWCTX";
    case DRM_IOCTL_AMDXDNA_CREATE_BO:
      return "DRM_IOCTL_AMDXDNA_CREATE_BO";
    case DRM_IOCTL_AMDXDNA_GET_BO_INFO:
      return "DRM_IOCTL_AMDXDNA_GET_BO_INFO";
    case DRM_IOCTL_AMDXDNA_SYNC_BO:
      return "DRM_IOCTL_AMDXDNA_SYNC_BO";
    case DRM_IOCTL_AMDXDNA_EXEC_CMD:
      return "DRM_IOCTL_AMDXDNA_EXEC_CMD";
    case DRM_IOCTL_AMDXDNA_WAIT_CMD:
      return "DRM_IOCTL_AMDXDNA_WAIT_CMD";
    case DRM_IOCTL_AMDXDNA_GET_INFO:
      return "DRM_IOCTL_AMDXDNA_GET_INFO";
    case DRM_IOCTL_AMDXDNA_SET_STATE:
      return "DRM_IOCTL_AMDXDNA_SET_STATE";
    case DRM_IOCTL_GEM_CLOSE:
      return "DRM_IOCTL_GEM_CLOSE";
    case DRM_IOCTL_PRIME_HANDLE_TO_FD:
      return "DRM_IOCTL_PRIME_HANDLE_TO_FD";
    case DRM_IOCTL_PRIME_FD_TO_HANDLE:
      return "DRM_IOCTL_PRIME_FD_TO_HANDLE";
    case DRM_IOCTL_SYNCOBJ_CREATE:
      return "DRM_IOCTL_SYNCOBJ_CREATE";
    case DRM_IOCTL_SYNCOBJ_QUERY:
      return "DRM_IOCTL_SYNCOBJ_QUERY";
    case DRM_IOCTL_SYNCOBJ_DESTROY:
      return "DRM_IOCTL_SYNCOBJ_DESTROY";
    case DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD:
      return "DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD";
    case DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE:
      return "DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE";
    case DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL:
      return "DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL";
    case DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT:
      return "DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT";
  }

  return "UNKNOWN(" + std::to_string(cmd) + ")";
}

size_t bar_size(const std::string& dir, unsigned bar) {
  std::ifstream ifs(dir + "/resource");
  if (!ifs.good()) return 0;
  std::string line;
  for (unsigned i = 0; i <= bar; i++) {
    line.clear();
    std::getline(ifs, line);
  }
  long long start, end, meta;
  if (sscanf(line.c_str(), "0x%llx 0x%llx 0x%llx", &start, &end, &meta) != 3)
    return 0;
  return end - start + 1;
}

int get_render_value(const std::string& dir,
                     const std::string& devnode_prefix) {
  struct dirent* entry;
  DIR* dp;
  int instance_num = INVALID_ID;

  dp = opendir(dir.c_str());
  if (dp == nullptr) return instance_num;

  while ((entry = readdir(dp))) {
    std::string dirname{entry->d_name};
    if (dirname.compare(0, devnode_prefix.size(), devnode_prefix) == 0) {
      instance_num = std::stoi(dirname.substr(devnode_prefix.size()));
      break;
    }
  }

  closedir(dp);

  return instance_num;
}

bool is_admin() { return (getuid() == 0) || (geteuid() == 0); }

const size_t dev_mem_size = (64 << 20);

}  // namespace

namespace sysfs {

static constexpr const char* dev_root = "/sys/bus/pci/devices/";

static std::string get_name(const std::string& dir, const std::string& subdir) {
  std::string line;
  std::ifstream ifs(dir + "/" + subdir + "/name");

  if (ifs.is_open()) std::getline(ifs, line);

  return line;
}

// Helper to find subdevice directory name
// Assumption: all subdevice's sysfs directory name starts with subdevice name!!
static int get_subdev_dir_name(const std::string& dir,
                               const std::string& subDevName,
                               std::string& subdir) {
  DIR* dp;
  size_t sub_nm_sz = subDevName.size();

  subdir = "";
  if (subDevName.empty()) return 0;

  int ret = -ENOENT;
  dp = opendir(dir.c_str());
  if (dp) {
    struct dirent* entry;
    while ((entry = readdir(dp))) {
      std::string nm = get_name(dir, entry->d_name);
      if (!nm.empty()) {
        if (nm != subDevName) continue;
      } else if (strncmp(entry->d_name, subDevName.c_str(), sub_nm_sz) != 0 ||
                 entry->d_name[sub_nm_sz] != '.') {
        continue;
      }
      // found it
      subdir = entry->d_name;
      ret = 0;
      break;
    }
    closedir(dp);
  }

  return ret;
}

static std::string get_path(const std::string& name, const std::string& subdev,
                            const std::string& entry) {
  std::string subdir;
  if (get_subdev_dir_name(dev_root + name, subdev, subdir) != 0) return "";

  std::string path = dev_root;
  path += name;
  path += "/";
  path += subdir;
  path += "/";
  path += entry;
  return path;
}

static std::fstream open_path(const std::string& path, std::string& err,
                              bool write, bool binary) {
  std::fstream fs;
  std::ios::openmode mode = write ? std::ios::out : std::ios::in;

  if (binary) mode |= std::ios::binary;

  err.clear();
  fs.open(path, mode);
  if (!fs.is_open()) {
    std::stringstream ss;
    ss << "Failed to open " << path << " for " << (binary ? "binary " : "")
       << (write ? "writing" : "reading") << ": " << strerror(errno)
       << std::endl;
    err = ss.str();
  }
  return fs;
}

static std::fstream open(const std::string& name, const std::string& subdev,
                         const std::string& entry, std::string& err, bool write,
                         bool binary) {
  std::fstream fs;
  auto path = get_path(name, subdev, entry);

  if (path.empty()) {
    std::stringstream ss;
    ss << "Failed to find subdirectory for " << subdev << " under "
       << dev_root + name << std::endl;
    err = ss.str();
  } else {
    fs = open_path(path, err, write, binary);
  }

  return fs;
}

static void get(const std::string& name, const std::string& subdev,
                const std::string& entry, std::string& err,
                std::vector<std::string>& sv) {
  std::fstream fs = open(name, subdev, entry, err, false, false);
  if (!err.empty()) return;

  sv.clear();
  std::string line;
  while (std::getline(fs, line)) sv.push_back(line);
}

static void get(const std::string& name, const std::string& subdev,
                const std::string& entry, std::string& err,
                std::vector<uint64_t>& iv) {
  iv.clear();

  std::vector<std::string> sv;
  get(name, subdev, entry, err, sv);
  if (!err.empty()) return;

  for (auto& s : sv) {
    if (s.empty()) {
      std::stringstream ss;
      ss << "Reading " << get_path(name, subdev, entry) << ", ";
      ss << "can't convert empty string to integer" << std::endl;
      err = ss.str();
      break;
    }
    char* end = nullptr;
    auto n = std::strtoull(s.c_str(), &end, 0);
    if (*end != '\0') {
      std::stringstream ss;
      ss << "Reading " << get_path(name, subdev, entry) << ", ";
      ss << "failed to convert string to integer: " << s << std::endl;
      err = ss.str();
      break;
    }
    iv.push_back(n);
  }
}

static void get(const std::string& name, const std::string& subdev,
                const std::string& entry, std::string& err, std::string& s) {
  std::vector<std::string> sv;
  get(name, subdev, entry, err, sv);
  if (!sv.empty())
    s = sv[0];
  else
    s = "";  // default value
}

static void get(const std::string& name, const std::string& subdev,
                const std::string& entry, std::string& err,
                std::vector<char>& buf) {
  std::fstream fs = open(name, subdev, entry, err, false, true);
  if (!err.empty()) return;

  buf.clear();
  buf.insert(std::end(buf), std::istreambuf_iterator<char>(fs),
             std::istreambuf_iterator<char>());
}

static void put(const std::string& name, const std::string& subdev,
                const std::string& entry, std::string& err,
                const std::string& input) {
  std::fstream fs = open(name, subdev, entry, err, true, false);
  if (!err.empty()) return;
  fs << input;
  fs.close();  // flush and close, if either fails then stream failbit is set.
  if (!fs.good()) {
    std::stringstream ss;
    ss << "Failed to write " << get_path(name, subdev, entry) << ": "
       << strerror(errno) << std::endl;
    err = ss.str();
  }
}

static void put(const std::string& name, const std::string& subdev,
                const std::string& entry, std::string& err,
                const std::vector<char>& buf) {
  std::fstream fs = open(name, subdev, entry, err, true, true);
  if (!err.empty()) return;

  fs.write(buf.data(), buf.size());
  fs.close();  // flush and close, if either fails then stream failbit is set.
  if (!fs.good()) {
    std::stringstream ss;
    ss << "Failed to write " << get_path(name, subdev, entry) << ": "
       << strerror(errno) << std::endl;
    err = ss.str();
  }
}

static void put(const std::string& name, const std::string& subdev,
                const std::string& entry, std::string& err,
                const unsigned int& input) {
  std::fstream fs = open(name, subdev, entry, err, true, false);
  if (!err.empty()) return;
  fs << input;
  fs.close();  // flush and close, if either fails then stream failbit is set.
  if (!fs.good()) {
    std::stringstream ss;
    ss << "Failed to write " << get_path(name, subdev, entry) << ": "
       << strerror(errno) << std::endl;
    err = ss.str();
  }
}

}  // namespace sysfs

namespace shim_xdna {

void pdev::sysfs_get(const std::string& subdev, const std::string& entry,
                     std::string& err, std::vector<uint64_t>& ret) const {
  sysfs::get(m_sysfs_name, subdev, entry, err, ret);
}

pdev::pdev(std::shared_ptr<const drv> driver, std::string sysfs_name)
    : m_driver(std::move(driver)), m_sysfs_name(std::move(sysfs_name)) {
  std::string err;

  if (sscanf(m_sysfs_name.c_str(), "%hx:%hx:%hx.%hx", &m_domain, &m_bus, &m_dev,
             &m_func) < 4)
    llvm::report_fatal_error(llvm::Twine(m_sysfs_name) + " is not valid BDF");

  m_is_mgmt = !m_driver->is_user();

  if (m_is_mgmt) {
    sysfs_get("", "instance", err, m_instance,
              static_cast<uint32_t>(INVALID_ID));
  } else {
    m_instance = get_render_value(
        sysfs::dev_root + m_sysfs_name + "/" + m_driver->sysfs_dev_node_dir(),
        m_driver->dev_node_prefix());
  }

  sysfs_get<int>("", "userbar", err, m_user_bar, 0);
  m_user_bar_size = bar_size(sysfs::dev_root + m_sysfs_name, m_user_bar);
  sysfs_get<bool>("", "ready", err, m_is_ready, false);
  m_user_bar_map = reinterpret_cast<char*>(MAP_FAILED);
  m_is_ready = true;  // We're always ready.
}

pdev::~pdev() {
  if (m_dev_fd != -1) shim_debug("Device node fd leaked!! fd=%d", m_dev_fd);
}

std::string pdev::get_subdev_path(const std::string& subdev, uint idx) const {
  // Main devfs path
  if (subdev.empty()) {
    std::string instStr = std::to_string(m_instance);
    std::string prefixStr = "/dev/";
    prefixStr += m_driver->dev_node_dir() + "/" + m_driver->dev_node_prefix();
    return prefixStr + instStr;
  }

  llvm::report_fatal_error("subdev path not supported");
}

int pdev::open(const std::string& subdev, uint32_t idx, int flag) const {
  if (m_is_mgmt && !::is_admin())
    llvm::report_fatal_error("Root privileges required");

  std::string devfs = get_subdev_path(subdev, idx);
  return ::open(devfs.c_str(), flag);
}

int pdev::open(const std::string& subdev, int flag) const {
  return open(subdev, 0, flag);
}

void pdev::open() const {
  int fd;
  const std::lock_guard<std::mutex> lock(m_lock);

  if (m_dev_users == 0) {
    fd = pdev::open("", O_RDWR);
    if (fd < 0)
      shim_err(EINVAL, "Failed to open KMQ device");
    else
      shim_debug("Device opened, fd=%d", fd);
    // Publish the fd for other threads to use.
    m_dev_fd = fd;
  }
  ++m_dev_users;
}

void pdev::close() const {
  int fd;
  const std::lock_guard<std::mutex> lock(m_lock);

  --m_dev_users;
  if (m_dev_users == 0) {
    on_last_close();

    // Stop new users of the fd from other threads.
    fd = m_dev_fd;
    m_dev_fd = -1;
    // Kernel will wait for existing users to quit.
    ::close(fd);
    shim_debug("Device closed, fd=%d", fd);
  }
}

void pdev::ioctl(unsigned long cmd, void* arg) const {
  if (::ioctl(m_dev_fd, cmd, arg) == -1)
    shim_err(errno, "%s IOCTL failed", ioctl_cmd2name(cmd).c_str());
}

void* pdev::mmap(void* addr, size_t len, int prot, int flags,
                 off_t offset) const {
  void* ret = ::mmap(addr, len, prot, flags, m_dev_fd, offset);

  if (ret == reinterpret_cast<void*>(-1))
    shim_err(errno,
             "mmap(addr=%p, len=%ld, prot=%d, flags=%d, offset=%ld) failed",
             addr, len, prot, flags, offset);
  return ret;
}

void pdev::munmap(void* addr, size_t len) const { ::munmap(addr, len); }

std::shared_ptr<device> pdev::create_device(void* handle) const {
  auto dev = std::make_shared<device>(*this, handle);
  // Alloc device memory on first device creation.
  // No locking is needed since driver will ensure only one heap BO is
  // created.
  if (m_dev_heap_bo == nullptr)
    m_dev_heap_bo =
        std::make_unique<bo>(*dev, dev_mem_size, AMDXDNA_BO_DEV_HEAP);
  return dev;
}

void pdev::on_last_close() const { m_dev_heap_bo.reset(); }

}  // namespace shim_xdna
