@echo off
rem ============================================================
rem FluxAgent overnight one-shot validation (cmd.exe only, no PowerShell)
rem Stages: full regression -> targeted eval (deterministic + real model
rem         + LLM judge) -> markdown summaries
rem Any stage failure stops immediately (exit code propagated).
rem All artifacts stay inside the project folder.
rem ============================================================
setlocal
cd /d "%~dp0"

call conda activate flux
if errorlevel 1 (
    echo [FATAL] conda activate flux failed
    exit /b 1
)

if not exist artifacts mkdir artifacts

echo [1/4] full regression: pytest tests --ignore=tests/eval
python -m pytest tests -q --ignore=tests/eval > artifacts\pytest_regression.log 2>&1
if errorlevel 1 (
    echo [FATAL] full regression failed, see artifacts\pytest_regression.log
    exit /b 1
)

echo [2/4] capability control-plane unit and integration tests
python -m pytest tests/test_capability_registry.py tests/test_hybrid_router.py tests/test_context_compiler.py tests/test_execution_gateway.py tests/test_capability_agent_integration.py -q > artifacts\pytest_capability.log 2>&1
if errorlevel 1 (
    echo [FATAL] capability tests failed, see artifacts\pytest_capability.log
    exit /b 1
)

echo [3/4] targeted eval: deterministic + real model + LLM judge
python -m fluxagent.experiments.post_change_eval --output artifacts --real --judge > artifacts\post_change_eval.log 2>&1
if errorlevel 1 (
    echo [FATAL] targeted eval has failing cases, see artifacts\post_change_eval.log and artifacts\post_change_eval.json
    exit /b 1
)

echo [4/4] generate markdown summaries and unused-files list
python -m fluxagent.experiments.write_summaries --output .
if errorlevel 1 (
    echo [FATAL] summary generation failed
    exit /b 1
)

echo [DONE] all stages passed. artifacts\ + CAPABILITY_ROUTING_*.md in project root
exit /b 0
