#!/bin/bash
# "After" profile: code <after rev> paired against <before rev> on the same clips, under the lock:
#   python -m bench.heavy --lane cpu --need 30 --floor 12 -- bash bench/opt/run_profile_after.sh <before> <after> <out dir>
set -u
A="$1"; B="$2"; OUT="$3"
mkdir -p "$OUT"
P="python -m bench.opt.profile"
s() { echo "{\"name\":\"$1\",\"code\":\"$2\",\"precision\":\"$3\"}"; }
log() { echo "$(date '+%H:%M:%S') load $(sysctl -n vm.loadavg) $*"; }
log start
$P latency --configs "$(s before-fp32 $A fp32),$(s after-fp32 $B fp32)" --langs all --threads 1,2,4 --repeats 2 --out "$OUT/opt_profile_after_fp32.json"; log fp32 latency done
$P latency --configs "$(s before-int8 $A int8),$(s after-int8 $B int8)" --langs all --threads 1,2,4 --repeats 2 --out "$OUT/opt_profile_after_int8.json"; log int8 latency done
$P rss --configs "$(s after-fp32 $B fp32),$(s after-int8 $B int8)" --langs en,ja,zh --threads 1,4 --n 20 --out "$OUT/opt_rss_after.json"; log rss done
$P footprint --config "$(s after-fp32 $B fp32)" --out "$OUT/opt_footprint_after_fp32.json"; log footprint fp32 done
$P footprint --config "$(s after-int8 $B int8)" --out "$OUT/opt_footprint_after_int8.json"; log footprint int8 done
$P firstload --configs "$(s before-fp32 $A fp32),$(s after-fp32 $B fp32),$(s before-int8 $A int8),$(s after-int8 $B int8)" --lang en --threads 4 --out "$OUT/opt_first_load.json"; log firstload done
C() { python -m bench.opt.candidates spec "$1" --code "$A"; }
CF="$(s fp32 $A fp32),$(C f16store-fp32),$(C nbits4-fp32)"
$P latency --configs "$CF" --langs en,zh --threads 1,4 --n 10 --repeats 3 --out "$OUT/opt_profile_cand_fp32b.json"; log cand fp32b latency done
$P rss --configs "$CF" --langs en --threads 1,4 --n 10 --out "$OUT/opt_rss_cand_fp32b.json"; log cand fp32b rss done
CI="$(s int8 $A int8),$(C int8x-embed+out),$(C int8x-embed+enc0+out),$(C slim-int8),$(C ortopt-ext-int8),$(C ortopt-basic-int8)"
$P latency --configs "$CI" --langs en,zh --threads 1,4 --n 10 --repeats 3 --out "$OUT/opt_profile_cand_int8.json"; log cand int8 latency done
$P rss --configs "$CI" --langs en --threads 1,4 --n 10 --out "$OUT/opt_rss_cand_int8.json"; log cand int8 rss done
log all done
