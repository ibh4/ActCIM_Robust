# ActCIM-Robust Submission Checklist

## Code Package
- [x] src/ directory complete (53 Python files, no __pycache__)
- [x] configs/ directory with 11 YAML files
- [x] scripts/ directory with 9 scripts
- [x] tests/ directory with 8 test files
- [x] requirements.txt
- [x] pyproject.toml
- [x] README.md

## Reports
- [x] ActCIM_Robust_technical_report.md
- [x] ActCIM_Robust_paper_draft.md
- [x] ppt_page_content.md
- [x] video_script.md
- [x] demo_runbook.md

## Experimental Results
- [x] all_methods_alpha_sweep.csv
- [x] all_methods_alpha_sweep.json
- [x] robustness_main_results.csv
- [x] fixed_nat_alpha_sweep.csv

## Figures
- [x] 01_accuracy_vs_alpha_all_methods.png
- [x] 05_worst_case_accuracy_comparison.png
- [x] 06_aurc_all_comparison.png
- [x] 08_clean_accuracy_vs_worst_accuracy.png

## Configs for Reproduction
- [x] baseline_fast.yaml
- [x] fixed_nat_positive.yaml
- [x] random_nat.yaml
- [x] sgr_nat.yaml

## Quality Checks
- [x] All P0 experiments completed
- [x] Unit tests pass (26/26)
- [x] Alpha sweep analysis complete
- [x] Layer sensitivity analysis complete
- [x] Error accumulation analysis complete
- [x] Random-NAT training complete
- [x] SGR-NAT training complete
- [x] All figures generated (PNG+PDF, 300 DPI)
- [x] Technical report complete
- [x] Paper draft complete
- [x] PPT content complete
- [x] Video script complete
- [x] Demo runbook complete
- [x] README complete
- [x] No hardcoded paths in submission
- [x] No invalid files in submission
- [x] No fabricated data
- [x] SHA256 checksums generated

## Checkpoints (Not Included - Too Large)
- [x] Clean baseline (seed 42): results/baseline/seed_42/best.pt (85MB)
- [x] Fixed-NAT (seed 42): results/fixed_nat/fixed_nat/seed_42/best.pt (85MB)
- [x] Reproduction instructions in README

## Submission Package Summary
- Total files: See SHA256SUMS.txt
- All items verified: YES
- Ready for submission: YES
