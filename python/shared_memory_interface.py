import mmap
import os
import struct
import time

import numpy as np
import posix_ipc


class SharedMemoryInterface:
    def __init__(self):
        # Wait for the C++ program to create the shared memory
        max_retries = 10
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Open shared memory objects
                self.shm = posix_ipc.SharedMemory("orbslam_shared_mem")
                self.state_shm = posix_ipc.SharedMemory("orbslam_state")

                # Open semaphores
                self.mutex = posix_ipc.Semaphore("orbslam_mutex")
                self.data_ready = posix_ipc.Semaphore("orbslam_data_ready")
                self.data_processed = posix_ipc.Semaphore(
                    "orbslam_data_processed")

                # Map the memory
                self.mapfile = mmap.mmap(self.shm.fd, self.shm.size)
                self.state_mapfile = mmap.mmap(
                    self.state_shm.fd, self.state_shm.size)

                self._set_processing_complete(True)
                self._set_new_data_flag(False)

                # Signal processing complete
                self.data_processed.release()

                break
            except posix_ipc.ExistentialError:
                retry_count += 1
                time.sleep(0.5)
                if retry_count == max_retries:
                    raise RuntimeError(
                        "Failed to connect to shared memory after multiple attempts"
                    )

    def read_data(self):
        # Wait for new data
        self.data_ready.acquire()

        # Acquire mutex
        self.mutex.acquire()

        try:

            # Read state information
            # Offsets for num_points, image_size, image_width, and image_height considering size_t alignment
            # 'Q' is for size_t (8 bytes) on 64-bit systems
            num_points = self._read_state_value("Q", 8)
            image_size = self._read_state_value("Q", 16)  # 'Q' for size_t
            image_width = self._read_state_value("i", 24)  # 'i' for int (4 bytes)
            image_height = self._read_state_value("i", 28)  # 'i' for int (4 bytes)

            # print(f"num points: {num_points}")
            # print(f"image size: {image_size}")
            # print(f"image width: {image_width}")
            # print(f"image height: {image_height}")

            # Read points
            points = []
            point_offset = 0
            for i in range(num_points):
                point_data = struct.unpack(
                    "=fffq", self.mapfile[point_offset: point_offset + 20]
                )
                points.append(
                    {
                        "x": point_data[0],
                        "y": point_data[1],
                        "z": point_data[2],
                        "id": point_data[3],
                    }
                )
                point_offset += 20

            # Read image
            image_data = np.frombuffer(
                self.mapfile[point_offset: point_offset +
                    image_size], dtype=np.uint8
            ).reshape((image_height, image_width, -1))
            # print(image_data)

            # Read camera pose
            camera_pos = np.array(
                struct.unpack(
                    "fff",
                    # 3 floats
                    self.state_mapfile[32:44],
                )
            )
            camera_rot = np.array(
                struct.unpack(
                    "fffffffff",
                    # 9 floats
                    self.state_mapfile[44:80],
                )
            ).reshape(3, 3)

            # Signal that we've processed the data
            self._set_processing_complete(True)
            self._set_new_data_flag(False)

        finally:
            # Release mutex
            self.mutex.release()

        # Signal processing complete
        self.data_processed.release()

        return points, image_data, camera_pos, camera_rot

    def _get_new_data_flag(self):
        return bool(struct.unpack("?", self.state_mapfile[72:73])[0])

    def _set_new_data_flag(self, value):
        self.state_mapfile[72:73] = struct.pack("?", value)

    def _set_processing_complete(self, value):
        self.state_mapfile[73:74] = struct.pack("?", value)

    def _read_state_value(self, format_char, offset):
        size = struct.calcsize(format_char)
        return struct.unpack(format_char, self.state_mapfile[offset : offset + size])[0]

    def __del__(self):
        try:
            self.mapfile.close()
            self.state_mapfile.close()
            self.shm.close_fd()
            self.state_shm.close_fd()
            self.mutex.close()
            self.data_ready.close()
            self.data_processed.close()
        except:
            pass
