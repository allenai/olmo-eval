#!/bin/sh

echo "########## CONTAINER DIAGNOSTICS ##########"
echo "Collected at: $(date -Iseconds 2>/dev/null || date)"

echo ""
echo "========== SYSTEM INFO =========="
hostname 2>/dev/null
uname -a 2>/dev/null

echo ""
echo "========== MEMORY DETAILED =========="
if [ -f /sys/fs/cgroup/memory.current ]; then
    echo "memory.current: $(cat /sys/fs/cgroup/memory.current)"
    echo "memory.max: $(cat /sys/fs/cgroup/memory.max)"
    echo "memory.swap.current: $(cat /sys/fs/cgroup/memory.swap.current 2>/dev/null || echo N/A)"
    echo "--- memory.events ---"
    cat /sys/fs/cgroup/memory.events 2>/dev/null
    echo "--- memory.pressure ---"
    cat /sys/fs/cgroup/memory.pressure 2>/dev/null
elif [ -f /sys/fs/cgroup/memory/memory.usage_in_bytes ]; then
    echo "usage: $(cat /sys/fs/cgroup/memory/memory.usage_in_bytes)"
    echo "limit: $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"
    echo "max_usage: $(cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes 2>/dev/null)"
    echo "failcnt: $(cat /sys/fs/cgroup/memory/memory.failcnt 2>/dev/null)"
    cat /sys/fs/cgroup/memory/memory.oom_control 2>/dev/null
fi

echo ""
echo "========== CPU DETAILED =========="
if [ -f /sys/fs/cgroup/cpu.stat ]; then
    cat /sys/fs/cgroup/cpu.stat 2>/dev/null
    echo "cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
    cat /sys/fs/cgroup/cpu.pressure 2>/dev/null
elif [ -f /sys/fs/cgroup/cpu/cpu.stat ]; then
    cat /sys/fs/cgroup/cpu/cpu.stat 2>/dev/null
fi
cat /proc/loadavg 2>/dev/null

echo ""
echo "========== DISK =========="
df -h 2>/dev/null
df -i 2>/dev/null
cat /sys/fs/cgroup/io.pressure 2>/dev/null

echo ""
echo "========== FILE DESCRIPTORS =========="
cat /proc/sys/fs/file-nr 2>/dev/null
echo "--- per-process (top 10) ---"
for pid in $(ls -d /proc/[0-9]* 2>/dev/null | head -30 | xargs -n1 basename); do
    count=$(ls /proc/$pid/fd 2>/dev/null | wc -l)
    comm=$(cat /proc/$pid/comm 2>/dev/null)
    [ "$count" -gt 20 ] && echo "$pid ($comm): $count"
done 2>/dev/null | sort -t: -k2 -rn | head -10

echo ""
echo "========== PROCESSES =========="
echo "Total: $(ps -e --no-headers 2>/dev/null | wc -l)"
echo "Zombies: $(ps -eo stat 2>/dev/null | grep -c '^Z')"
echo "D-state: $(ps -eo stat 2>/dev/null | grep -c '^D')"
if [ -f /sys/fs/cgroup/pids.current ]; then
    echo "PIDs: $(cat /sys/fs/cgroup/pids.current)/$(cat /sys/fs/cgroup/pids.max)"
fi
echo "--- all processes ---"
ps auxf 2>/dev/null || ps aux 2>/dev/null
echo "--- D-state with wait channel ---"
ps -eo pid,stat,wchan:32,comm 2>/dev/null | grep " D"
echo "--- zombies ---"
ps -eo pid,ppid,stat,comm 2>/dev/null | grep " Z"

echo ""
echo "========== THREADS =========="
ps -eo pid,nlwp,comm --sort=-nlwp 2>/dev/null | head -10

echo ""
echo "========== NETWORK =========="
ss -s 2>/dev/null
echo "--- listening ---"
ss -tlnp 2>/dev/null | head -10
echo "--- established ---"
ss -tnp 2>/dev/null | head -15
echo "--- socket states ---"
ss -tan 2>/dev/null | awk 'NR>1{print $1}' | sort | uniq -c | sort -rn
echo "--- DNS test ---"
timeout 2 getent hosts google.com 2>&1 || echo "DNS failed"

echo ""
echo "========== KERNEL MESSAGES =========="
dmesg 2>/dev/null | tail -30 || echo "dmesg not available"

echo ""
echo "========== STACK TRACES =========="
for pid in $(ps -eo pid --no-headers 2>/dev/null | head -15); do
    if [ -f /proc/$pid/stack ]; then
        comm=$(cat /proc/$pid/comm 2>/dev/null)
        echo "--- PID $pid ($comm) ---"
        head -8 /proc/$pid/stack 2>/dev/null
    fi
done 2>/dev/null

echo "########## END DIAGNOSTICS ##########"
