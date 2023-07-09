start_time=$(date +%s%3N)

# 1
python3 inference_member_v2.py&
# 2
python3 inference_conDamage.py&&
# 3
python3 bbox_intersection.py;
# 4
python3 inference_fastener.py;
# 5
python3 inference_railDamage_v2.py

end_time=$(date +%s%3N)

total_time=$(( (end_time - start_time) / 1000 ))

minutes=$(( total_time / 60 ))
seconds=$(( total_time % 60 ))

time_per_image=$((total_time*1000 / 3508))

echo "Total time: ${minutes} min ${seconds} sec"
echo "Time per image: ${time_per_image} ms/image"