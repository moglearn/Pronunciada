#!/bin/bash
# Profile of record for the optimization report, on one commit, run under the machine-wide lock:
#   python -m bench.heavy --lane cpu --need 30 --floor 12 -- bash bench/opt/run_profile_of_record.sh <rev> <out dir>
# 1. baseline latency: fp32 and int8, all ten languages, 1 / 2 / 4 threads, 20 practice clips, 3 rounds
# 2. so762 latency (English learner clips, 3.4 s): fp32 and int8, 1 / 4 threads, 200 clips, 1 round
# 3. peak RSS: fp32 and int8, en / ja / zh, 1 / 4 threads, 20 clips
# 4. footprint: fp32 and int8, all languages
# 5. candidate latency: identical ONNX Runtime option sets and lossless files against their default,
#    en and zh, 1 / 4 threads, 10 clips, 3 rounds; RSS of the same
set -u
REV="$1"; OUT="$2"
mkdir -p "$OUT"
P="python -m bench.opt.profile"
C() { python -m bench.opt.candidates spec "$1" --code "$REV"; }
F32="{\"name\":\"fp32\",\"code\":\"$REV\",\"precision\":\"fp32\"}"
I8="{\"name\":\"int8\",\"code\":\"$REV\",\"precision\":\"int8\"}"
log() { echo "$(date '+%H:%M:%S') load $(sysctl -n vm.loadavg) $*"; }
log start
$P latency --configs "$F32,$I8" --langs all --threads 1,2,4 --repeats 3 --out "$OUT/opt_profile_baseline.json"; log latency done
$P latency --configs "$F32,$I8" --set so762 --langs en --threads 1,4 --repeats 1 --out "$OUT/opt_profile_so762.json"; log so762 done
$P rss --configs "$F32,$I8" --langs en,ja,zh --threads 1,4 --n 20 --out "$OUT/opt_rss_baseline.json"; log rss done
$P footprint --config "$F32" --out "$OUT/opt_footprint_fp32.json"; log footprint fp32 done
$P footprint --config "$I8" --out "$OUT/opt_footprint_int8.json"; log footprint int8 done
CF="$F32,$(C ort-arena-off),$(C ort-mempattern-on),$(C ort-nospin),$(C ort-ext-level),$(C slim-fp32),$(C ortopt-ext-fp32),$(C ortopt-basic-fp32),$(C prep-fp32)"
$P latency --configs "$CF" --langs en,zh --threads 1,4 --n 10 --repeats 3 --out "$OUT/opt_profile_cand_fp32.json"; log cand fp32 latency done
$P rss --configs "$CF" --langs en --threads 1,4 --n 10 --out "$OUT/opt_rss_cand_fp32.json"; log cand fp32 rss done
log all done
