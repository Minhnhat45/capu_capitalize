#! /bin/sh
#
# run_infer.sh
# Copyright (C) 2022 nhat-l <nhat-l@rd04>
#
# Distributed under terms of the MIT license.
#


curl -i -H "Content-Type: application/json" -X POST -d '{"input_string":"ngày hôm nay quảng ninh nắng nóng độ ẩm cao"}' http://localhost:4445/capu
