#!/bin/sh
# Internal container monitor - runs as background process
# Writes to /sandbox_logs/ which is volume-mounted from host
#
# Output files:
#   stats.log    - Human-readable metrics history (appended)
#   metrics.json - JSON snapshot of latest metrics (overwritten)

LOGFILE="/sandbox_logs/stats.log"
JSONFILE="/sandbox_logs/metrics.json"
INTERVAL=5

# Ensure we can write
if ! touch "$LOGFILE"; then
    echo "Cannot write to $LOGFILE" >&2
    exit 1
fi

while true; do
    {
        echo "=== $(date -Iseconds 2>/dev/null || date) ==="

        echo "--- MEMORY ---"
        # cgroups v2
        if [ -f /sys/fs/cgroup/memory.current ]; then
            curr=$(cat /sys/fs/cgroup/memory.current 2>/dev/null)
            max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)
            echo "current_bytes: $curr"
            echo "max_bytes: $max"
            if [ "$max" != "max" ] && [ -n "$curr" ] && [ -n "$max" ] && [ "$max" -gt 0 ] 2>/dev/null; then
                pct=$((curr * 100 / max))
                echo "usage_percent: ${pct}%"
            fi
            # OOM events - critical for diagnosing kills
            grep -E "^(oom|oom_kill)" /sys/fs/cgroup/memory.events 2>/dev/null
            # Memory pressure
            head -1 /sys/fs/cgroup/memory.pressure 2>/dev/null
        # cgroups v1
        elif [ -f /sys/fs/cgroup/memory/memory.usage_in_bytes ]; then
            echo "usage_bytes: $(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null)"
            echo "limit_bytes: $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)"
            echo "failcnt: $(cat /sys/fs/cgroup/memory/memory.failcnt 2>/dev/null)"
            cat /sys/fs/cgroup/memory/memory.oom_control 2>/dev/null
        fi

        echo "--- CPU ---"
        # cgroups v2
        if [ -f /sys/fs/cgroup/cpu.stat ]; then
            # nr_throttled and throttled_usec show if we're being throttled
            grep -E "^(usage_usec|nr_throttled|throttled_usec)" /sys/fs/cgroup/cpu.stat 2>/dev/null
            head -1 /sys/fs/cgroup/cpu.pressure 2>/dev/null
        # cgroups v1
        elif [ -f /sys/fs/cgroup/cpu/cpu.stat ]; then
            cat /sys/fs/cgroup/cpu/cpu.stat 2>/dev/null
        fi
        cat /proc/loadavg 2>/dev/null

        echo "--- DISK ---"
        df -h / /tmp 2>/dev/null | tail -n +2
        df -i / /tmp 2>/dev/null | tail -n +2

        echo "--- IO PRESSURE ---"
        head -1 /sys/fs/cgroup/io.pressure 2>/dev/null || echo "N/A"

        echo "--- FILE DESCRIPTORS ---"
        cat /proc/sys/fs/file-nr 2>/dev/null | awk '{print "fds: " $1 "/" $3}'

        echo "--- PROCESSES ---"
        total=$(ps -e --no-headers 2>/dev/null | wc -l)
        zombies=$(ps -eo stat 2>/dev/null | grep -c "^Z" || echo 0)
        dstate=$(ps -eo stat 2>/dev/null | grep -c "^D" || echo 0)
        echo "total: $total, zombies: $zombies, D-state: $dstate"
        # PID limits
        if [ -f /sys/fs/cgroup/pids.current ]; then
            echo "pids: $(cat /sys/fs/cgroup/pids.current)/$(cat /sys/fs/cgroup/pids.max 2>/dev/null)"
        fi

        echo "--- NETWORK ---"
        ss -s 2>/dev/null | grep -E "^(TCP|UDP)" | head -2 || echo "N/A"

        echo "--- TOP PROCS (mem) ---"
        ps -eo pid,stat,%mem,%cpu,comm --sort=-%mem 2>/dev/null | head -6

        echo "--- D-STATE PROCS ---"
        ps -eo pid,wchan:20,comm 2>/dev/null | grep -v "WCHAN" | while read pid wchan comm; do
            state=$(cat /proc/$pid/stat 2>/dev/null | awk '{print $3}')
            [ "$state" = "D" ] && echo "$pid $wchan $comm"
        done | head -5

        echo ""
    } >> "$LOGFILE" 2>&1

    # Write JSON snapshot for programmatic consumption
    {
        echo "{"
        echo "  \"timestamp\": \"$(date -Iseconds 2>/dev/null || date)\","

        # Memory
        if [ -f /sys/fs/cgroup/memory.current ]; then
            curr=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
            max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo "max")
            oom=$(grep "^oom " /sys/fs/cgroup/memory.events 2>/dev/null | awk '{print $2}')
            oom_kill=$(grep "^oom_kill " /sys/fs/cgroup/memory.events 2>/dev/null | awk '{print $2}')
            echo "  \"memory\": {\"current_bytes\": $curr, \"max_bytes\": \"$max\", \"oom_events\": ${oom:-0}, \"oom_kills\": ${oom_kill:-0}},"
        elif [ -f /sys/fs/cgroup/memory/memory.usage_in_bytes ]; then
            curr=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0)
            max=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo 0)
            failcnt=$(cat /sys/fs/cgroup/memory/memory.failcnt 2>/dev/null || echo 0)
            echo "  \"memory\": {\"current_bytes\": $curr, \"max_bytes\": $max, \"failcnt\": $failcnt},"
        fi

        # CPU
        if [ -f /sys/fs/cgroup/cpu.stat ]; then
            throttled=$(grep "^nr_throttled " /sys/fs/cgroup/cpu.stat 2>/dev/null | awk '{print $2}')
            echo "  \"cpu\": {\"throttled_count\": ${throttled:-0}},"
        fi

        # Disk
        root_use=$(df / 2>/dev/null | tail -1 | awk '{print $5}' | tr -d '%')
        tmp_use=$(df /tmp 2>/dev/null | tail -1 | awk '{print $5}' | tr -d '%')
        root_avail=$(df / 2>/dev/null | tail -1 | awk '{print $4}')
        tmp_avail=$(df /tmp 2>/dev/null | tail -1 | awk '{print $4}')
        echo "  \"disk\": {\"root_percent\": ${root_use:-0}, \"tmp_percent\": ${tmp_use:-0}, \"root_avail\": \"${root_avail:-0}\", \"tmp_avail\": \"${tmp_avail:-0}\"},"

        # Processes
        total=$(ps -e --no-headers 2>/dev/null | wc -l)
        zombies=$(ps -eo stat 2>/dev/null | grep -c "^Z" || echo 0)
        dstate=$(ps -eo stat 2>/dev/null | grep -c "^D" || echo 0)
        echo "  \"processes\": {\"total\": $total, \"zombies\": $zombies, \"d_state\": $dstate}"

        echo "}"
    } > "$JSONFILE" 2>/dev/null

    # Keep log file from growing unbounded (keep last 1000 lines)
    if [ $(wc -l < "$LOGFILE" 2>/dev/null || echo 0) -gt 2000 ]; then
        tail -1000 "$LOGFILE" > "$LOGFILE.tmp" && mv "$LOGFILE.tmp" "$LOGFILE"
    fi

    sleep $INTERVAL
done
