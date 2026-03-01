#!/bin/sh
# Internal container monitor - runs as background process
# Writes to /sandbox_logs/ which is volume-mounted from host
#
# Output files:
#   stats.log - Human-readable metrics history (appended)

LOGFILE="/sandbox_logs/stats.log"
INTERVAL=5

# Helper to format bytes as human-readable
human_bytes() {
    bytes=$1
    if [ -z "$bytes" ] || [ "$bytes" = "max" ]; then
        echo "$bytes"
        return
    fi
    if [ "$bytes" -ge 1073741824 ] 2>/dev/null; then
        gb=$((bytes / 1073741824))
        mb=$(((bytes % 1073741824) / 10737418))
        echo "${gb}.${mb}G"
    elif [ "$bytes" -ge 1048576 ] 2>/dev/null; then
        echo "$((bytes / 1048576))M"
    elif [ "$bytes" -ge 1024 ] 2>/dev/null; then
        echo "$((bytes / 1024))K"
    else
        echo "${bytes}B"
    fi
}

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
            curr_h=$(human_bytes "$curr")
            max_h=$(human_bytes "$max")
            echo "usage: $curr_h / $max_h"
            if [ "$max" != "max" ] && [ -n "$curr" ] && [ -n "$max" ] && [ "$max" -gt 0 ] 2>/dev/null; then
                pct=$((curr * 100 / max))
                echo "percent: ${pct}%"
            fi
            # OOM events - critical for diagnosing kills
            oom=$(cat /sys/fs/cgroup/memory.events 2>/dev/null | grep "^oom " | awk '{print $2}')
            oom_kill=$(cat /sys/fs/cgroup/memory.events 2>/dev/null | grep "^oom_kill " | awk '{print $2}')
            if [ "${oom:-0}" -gt 0 ] || [ "${oom_kill:-0}" -gt 0 ]; then
                echo "OOM: $oom, OOM_KILL: $oom_kill"
            fi
        # cgroups v1
        elif [ -f /sys/fs/cgroup/memory/memory.usage_in_bytes ]; then
            curr=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null)
            max=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)
            echo "usage: $(human_bytes "$curr") / $(human_bytes "$max")"
            failcnt=$(cat /sys/fs/cgroup/memory/memory.failcnt 2>/dev/null)
            if [ "${failcnt:-0}" -gt 0 ]; then
                echo "failcnt: $failcnt"
            fi
        fi

        echo "--- CPU ---"
        # cgroups v2
        if [ -f /sys/fs/cgroup/cpu.stat ]; then
            throttled=$(grep "^nr_throttled " /sys/fs/cgroup/cpu.stat 2>/dev/null | awk '{print $2}')
            if [ "${throttled:-0}" -gt 0 ]; then
                throttled_usec=$(grep "^throttled_usec " /sys/fs/cgroup/cpu.stat 2>/dev/null | awk '{print $2}')
                echo "throttled: $throttled times (${throttled_usec}us)"
            fi
        fi
        # Load average (1, 5, 15 min + running/total procs)
        if [ -f /proc/loadavg ]; then
            read load1 load5 load15 procs _ < /proc/loadavg
            echo "load: $load1 $load5 $load15 ($procs)"
        fi

        echo "--- DISK ---"
        # Only show root filesystem once
        df -h / 2>/dev/null | awk 'NR==2 {print "space:", $3, "/", $2, "(" $5 " used)"}'
        df -i / 2>/dev/null | awk 'NR==2 {print "inodes:", $3, "/", $2, "(" $5 " used)"}'

        echo "--- PROCESSES ---"
        # Use /proc directly for portability (busybox ps differs from GNU ps)
        total=0
        zombies=0
        dstate=0
        for pid_dir in /proc/[0-9]*; do
            [ -d "$pid_dir" ] || continue
            total=$((total + 1))
            state=$(cat "$pid_dir/stat" 2>/dev/null | awk '{print $3}')
            case "$state" in
                Z) zombies=$((zombies + 1)) ;;
                D) dstate=$((dstate + 1)) ;;
            esac
        done
        echo "total: $total, zombies: $zombies, D-state: $dstate"
        # PID limits
        if [ -f /sys/fs/cgroup/pids.current ]; then
            pids_curr=$(cat /sys/fs/cgroup/pids.current 2>/dev/null)
            pids_max=$(cat /sys/fs/cgroup/pids.max 2>/dev/null)
            echo "pids: $pids_curr / $pids_max"
        fi

        echo "--- FILE DESCRIPTORS ---"
        if [ -f /proc/sys/fs/file-nr ]; then
            read allocated _ max < /proc/sys/fs/file-nr
            echo "fds: $allocated / $max"
        fi

        echo "--- TOP PROCS (by RSS) ---"
        # Parse /proc directly for portability
        {
            for pid_dir in /proc/[0-9]*; do
                [ -f "$pid_dir/stat" ] || continue
                pid=$(basename "$pid_dir")
                # Get RSS from statm (in pages, typically 4K)
                rss_pages=$(awk '{print $2}' "$pid_dir/statm" 2>/dev/null)
                [ -n "$rss_pages" ] || continue
                rss_kb=$((rss_pages * 4))
                # Get command name
                comm=$(cat "$pid_dir/comm" 2>/dev/null | head -1)
                [ -n "$comm" ] || continue
                echo "$rss_kb $pid $comm"
            done
        } | sort -rn | head -5 | while read rss_kb pid comm; do
            rss_h=$(human_bytes $((rss_kb * 1024)))
            printf "  %6s %s (%s)\n" "$rss_h" "$comm" "$pid"
        done

        # Only show D-state processes if any exist
        if [ "$dstate" -gt 0 ]; then
            echo "--- D-STATE PROCS ---"
            for pid_dir in /proc/[0-9]*; do
                [ -f "$pid_dir/stat" ] || continue
                state=$(awk '{print $3}' "$pid_dir/stat" 2>/dev/null)
                if [ "$state" = "D" ]; then
                    pid=$(basename "$pid_dir")
                    comm=$(cat "$pid_dir/comm" 2>/dev/null)
                    wchan=$(cat "$pid_dir/wchan" 2>/dev/null)
                    echo "  $pid $comm ($wchan)"
                fi
            done | head -5
        fi

        echo ""
    } >> "$LOGFILE" 2>&1

    # Keep log file from growing unbounded (keep last 1000 lines)
    if [ $(wc -l < "$LOGFILE" 2>/dev/null || echo 0) -gt 2000 ]; then
        tail -1000 "$LOGFILE" > "$LOGFILE.tmp" && mv "$LOGFILE.tmp" "$LOGFILE"
    fi

    sleep $INTERVAL
done
