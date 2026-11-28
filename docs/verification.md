# Verification and publication

The verified experiment used macOS on Apple Silicon, Python 3.13, seeded
CPU training and the versions in
[verified constraints](../configs/verified-constraints.txt).

| Check | Result |
|---|---|
| Download | Project CLI downloaded the official archive and matched the Zenodo MD5 |
| Data | Five schemas read successfully. The eligibility audit retained 1,554 intervals |
| Models | Mean, log-Ridge, Random Forest, Histogram Boosting and XGBoost fitted and scored |
| Evaluation | Five outer device folds, grouped inner tuning, chronological validation and 198 held-out sequences |
| Automated checks | 14 pytest cases passed, and Ruff reported no lint errors |
| Simulation | Charge bounds, constant drain and exact schedule crossing tested |
| Uncertainty | 40 sequence bootstrap refits, residual draws, coverage diagnostics and five sensitivity variants |
| Dashboard | All seven pages smoke-tested in Chrome with the trained artifact. Three screenshots captured |
| API | Request validation and shared engine checked with pytest. All six endpoints exercised over HTTP in Docker |
| Docker | Compose configuration parsed, arm64 image built and both services started. API and dashboard health checked |
| Hosted setup | Community Cloud uses the bundled Random Forest and an explorer table with relative times. The dashboard test exercises all seven pages from a fresh checkout. |
| Publication contents | Staged filenames and common secret patterns screened. Raw logs, full interval exports and environment files excluded. Only the serving model and reduced explorer table are bundled. |
| Licensing | Official Zenodo metadata identifies CC BY 4.0. The extension includes the MIT license text and original research attribution |

No eligible held-out discharge sequence ended in a recorded zero. Scenario
estimates and empirical discharge-rate coverage were verified. Empirical
time-to-empty accuracy remains unevaluated.

## Publish after approval

GitHub confirmed the authenticated account is `youssefrafik6`. After approval
for public visibility, run these commands from the local repository root.
Run the login command first if the session has expired.

```bash
# If authentication has expired:
# gh auth login --hostname github.com --git-protocol https --web
gh auth switch --hostname github.com --user youssefrafik6
gh auth setup-git --hostname github.com
gh api user --jq .login
gh repo create youssefrafik6/battery-life-ai --public --source=. --remote=origin --push \
  --description 'End-to-end smartphone battery life prediction and simulation platform using machine learning, continuous-time modeling, and Monte Carlo methods.'
gh repo view youssefrafik6/battery-life-ai --json url,visibility,defaultBranchRef
```

The authenticated login should be `youssefrafik6`. The final command
reports the URL and visibility confirmed by GitHub.
