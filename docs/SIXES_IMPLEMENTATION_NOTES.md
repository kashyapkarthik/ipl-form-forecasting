# Implementation clarifications

Recorded before the first full model run on 14 October 2026:

- Preseason selection uses the **player-level forecast averaged over past role exposure**, matching the planned 2027 deployment. Conditional-on-observed-role evaluation is a separate diagnostic using the same selected models. This resolves the protocol's distinction between the two preseason information sets without selecting on future data.
- The existing generic Beta-binomial optimiser is reused; outcome extraction and history construction are separate six-specific modules. No original boundary outcomes or derived boundary counts are used as six successes.
- Population denominators are exact pooled role counts. Only log scoring clips raw probabilities; Brier scoring retains exact zero/one probabilities.
