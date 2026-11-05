#!/bin/bash
# regress.sh -- the engine's regression suite: the checks listed in tests/regress.tsv, run on the models that are
# present (the others are skipped), one PASS / FAIL / SKIP line each, a summary, exit 1 if any failed.
#
#   tests/regress.sh [--quick] [--update] [name ...]
#
#   --quick    only the rows of tier "quick" (small models: a few minutes); otherwise all of them
#   --update   record each ppl row's result as its new baseline in regress.tsv (after a change that is meant to
#              move them: say why in the commit)
#   name ...   only the rows whose name starts with one of these
#
# Rows (tab-separated: tier, kind, name, model, args, expect, tol):
#   ppl   perplexity <model> <text> <args>: within tol percent of expect (the baseline); the text is $WIKITEXT
#         (default /data/hf/wikitext/wikitext-2-raw/wiki.test.raw, llama.cpp's scripts/get-wikitext-2.sh)
#   fv    forward_verify <model> <args = llama.cpp reference dump> with FORWARD_VERIFY_MIN_COS=expect: exit 0
#   pk    passkey <model> <text> <args> (a pass key found at every context length and depth): exit 0
#   mv    matvec_verify <model> <args = tensors>: exit 0
#   test  <args = a test binary and its arguments; the model, if any, is its first argument>: exit 0
# Paths may use $VARS (exported here: REFS). Runs from build/co, as the binaries expect; logs in $REGRESS_LOGS
# (default /tmp/regress-logs).
set -u
cd "$(dirname "$0")/../build/co" || exit 2
TSV=../../tests/regress.tsv
export REFS=${REFS:-/data/q4_0/refs}
W=${WIKITEXT:-/data/hf/wikitext/wikitext-2-raw/wiki.test.raw}
LOGS=${REGRESS_LOGS:-/tmp/regress-logs}
mkdir -p "$LOGS"
quick=0 update=0 names=()
for a in "$@"; do
    case $a in
        --quick) quick=1 ;;
        --update) update=1 ;;
        -*) echo "unknown option $a" >&2; exit 2 ;;
        *) names+=("$a") ;;
    esac
done
[ -f "$W" ] || { echo "no text for the perplexity rows: $W (set WIKITEXT)" >&2; }

pass=0 fail=0 skip=0
new=$(mktemp)
while IFS= read -r line; do
    IFS=$'\t' read -r tier kind name model args expect tol <<< "$line"
    run=1
    [[ -z $line || $line == \#* ]] && run=0
    if [ $run = 1 ]; then
        [ $quick = 1 ] && [ "$tier" != quick ] && run=0
        if [ ${#names[@]} -gt 0 ]; then
            hit=0
            for n in "${names[@]}"; do [[ $name == "$n"* ]] && hit=1; done
            [ $hit = 0 ] && run=0
        fi
    fi
    if [ $run = 0 ]; then echo "$line" >> "$new"; continue; fi
    raw_model=$model raw_args=$args
    model=$(eval echo "$model") args=$(eval echo "$args")
    log="$LOGS/$name.log"
    if [ "$model" != - ] && [ ! -f "$model" ]; then
        printf "SKIP  %-28s no %s\n" "$name" "$model"; skip=$((skip + 1)); echo "$line" >> "$new"; continue
    fi
    t0=$SECONDS
    case $kind in
        ppl)
            if [ ! -f "$W" ]; then
                printf "SKIP  %-28s no %s\n" "$name" "$W"; skip=$((skip + 1)); echo "$line" >> "$new"; continue
            fi
            # shellcheck disable=SC2086
            ../perplexity "$model" "$W" $args > "$log" 2>&1
            got=$(sed -n 's/^perplexity: \([0-9.]*\) +\/-.*/\1/p' "$log" | tail -1)
            if [ -z "$got" ]; then
                ok=0 msg="no result (see $log)"
            elif [ $update = 1 ]; then
                ok=1 msg="$got (baseline was $expect)"
                line=$(printf "%s\t%s\t%s\t%s\t%s\t%s\t%s" "$tier" "$kind" "$name" "$raw_model" "$raw_args" "$got" "$tol")
            else
                dev=$(awk -v g="$got" -v b="$expect" 'BEGIN { printf "%.3f", (g - b) / b * 100 }')
                ok=$(awk -v d="$dev" -v t="$tol" 'BEGIN { print (d < 0 ? -d : d) <= t ? 1 : 0 }')
                msg="$got vs $expect (${dev}%, tolerance ${tol}%)"
            fi ;;
        fv)
            FORWARD_VERIFY_MIN_COS=$expect ../forward_verify "$model" "$args" > "$log" 2>&1
            rc=$?; ok=$([ $rc = 0 ] && echo 1 || echo 0)
            msg="$(grep -E 'engine l_out' "$log" | sed 's/^ *\[[A-Z]*\] *//' | cut -c1-80) (exit $rc)" ;;
        pk)
            if [ ! -f "$W" ]; then
                printf "SKIP  %-28s no %s\n" "$name" "$W"; skip=$((skip + 1)); echo "$line" >> "$new"; continue
            fi
            # shellcheck disable=SC2086
            ../passkey "$model" "$W" $args > "$log" 2>&1
            rc=$?; ok=$([ $rc = 0 ] && echo 1 || echo 0)
            msg="$(tail -1 "$log") (exit $rc)" ;;
        mv)
            # shellcheck disable=SC2086
            ../matvec_verify "$model" $args > "$log" 2>&1
            rc=$?; ok=$([ $rc = 0 ] && echo 1 || echo 0)
            msg=$(tail -1 "$log") ;;
        test)
            set -- $args
            bin=$1; shift
            if [ "$model" = - ]; then ../$bin "$@" > "$log" 2>&1; else ../$bin "$model" "$@" > "$log" 2>&1; fi
            rc=$?; ok=$([ $rc = 0 ] && echo 1 || echo 0)
            msg="$(tail -1 "$log") (exit $rc)" ;;
        *) ok=0 msg="unknown kind $kind" ;;
    esac
    printf "%s  %-28s %s  [%ds]\n" "$([ "$ok" = 1 ] && echo PASS || echo FAIL)" "$name" "$msg" $((SECONDS - t0))
    if [ "$ok" = 1 ]; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
    echo "$line" >> "$new"
done < "$TSV"
if [ $update = 1 ]; then cp "$new" "$TSV"; echo "baselines updated in tests/regress.tsv"; fi
rm -f "$new"
echo "regress: $pass passed, $fail failed, $skip skipped"
[ $fail = 0 ]
