# GPU 1 上按显存 90 % 推算 n
./mapped_memory --device 1 --ratio 0.9

# 在 GPU 2 上显式 n = 150M
./mapped_memory -d 2 150000000
