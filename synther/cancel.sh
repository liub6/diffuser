#!/bin/bash

# 定义任务 ID 范围
start_id=4053987
end_id=4054356

# 循环遍历任务 ID 范围并取消任务
for job_id in $(seq $start_id $end_id); do
    scancel $job_id
done