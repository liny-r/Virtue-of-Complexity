# Speaking script: The Virtue of Complexity Everywhere (about 13.5 min, equities only)

Notation follows the lecture (Theoretical Foundations III): raw signals G_t, random features S_t, P features, complexity c = P/T, ridge penalty z, weights β̂, forecast μ̂_t, position π_t. The paper writes L where the lecture writes P.

Pace about 140 words per minute. Times are cumulative. Slide numbers are the deck's page numbers (1–14), then two backups.

| Speaker | Slides | Time |
|---|---|---|
| Luwei Cai | 1–4: question, notation, data | 0:00–3:30 |
| Hongjun Ding | 5–7: model, strategy, why it works | 3:30–7:00 |
| Yueqi Lin | 8–10: implementation, figures, replication | 7:00–10:15 |
| Hao Li | 11–14: baseline, placebo, critique, takeaways | 10:15–13:45 |

---

## Slide 1 — Title (0:00–0:15) — **Luwei Cai**

Good morning. We're Team 1: Luwei Cai, Hongjun Ding, Yueqi Lin and Hao Li. Our paper is "The Virtue of Complexity Everywhere" by Kelly, Malamud and Zhou. We'll cover the idea, the model, what we found when we replicated it, and where we think it is soft. We focus on equities.

## Slide 2 — The question (0:15–1:45)

The financial question is factor timing. Can we predict, out of sample, when a US equity factor will do well? Returns are noisy and histories are short. The paper's key case uses a training window of only twelve months.

The textbook answer is Occam's razor: keep the number of parameters P well below the number of observations T, or you fit noise.

The lecture gave us a different way to think about this, in Part III. There are two worlds. If your model is correctly specified, complexity is a pure statistical cost: you get classic double descent and never beat the oracle. But no return model is correctly specified. The true relationship is unknown and probably nonlinear. In that misspecified world, adding features removes approximation error, and with enough shrinkage the gain can outrun the extra noise.

The paper takes that to the data: use P much larger than T, on 159 equity factors. The claim is conditional. Sharpe rises with complexity when ridge shrinkage is large enough. It is not "bigger is always better".

## Slide 3 — Setup and notation (1:45–2:50)

Here is the notation, the same as in the lecture. G_t is the vector of raw signals at time t. It has d entries: fifteen Welch–Goyal macro variables in one study, two momentum signals in the other. R_{t+1} is next month's factor return, scaled by volatility known at time t.

S_t is the vector of P random features built from G_t. T is the number of months in the rolling training window: twelve, sixty or one hundred twenty. Complexity is c equals P over T. When c equals one, the number of features equals the number of observations, the interpolation boundary. z is the ridge penalty. And β̂ is the only thing that is learned: P weights.

Two points. Raising P adds transformations of the same d signals. It does not add any new financial data. And there are two separate studies, A with Welch–Goyal and B with momentum. They are never combined. In each, the model is estimated separately for each of the 159 factors.

## Slide 4 — Data (2:50–3:30)

The universe is six Fama–French factors plus 153 JKP characteristic factors, monthly, equities only. The paper also uses futures and Treasuries; we do not.

Everything is standardised using past data only: returns by their trailing twelve-month root mean square, signals by an expanding root mean square with at least 36 months, clipped to plus or minus five.

As a data check, we recomputed mean, volatility and Sharpe for the factors and compared with the paper's Table 1. The correlations are 0.996, 0.9995 and 0.997. So we are on the same data.

**[Handoff]** "Hongjun will take you through the model."

## Slide 5 — Model (3:30–5:00) — **Hongjun Ding**

Now the model. There are two parts, and it's important to be clear about which is fixed and which is learned.

The features are fixed. We draw random vectors ω_i from a standard normal in d dimensions, once. For each, we compute the sine and the cosine of γ times ω_i transpose G_t. So each ω_i gives two features, and stacking them gives S_t with P entries. We use seven bandwidths γ, from 0.1 up to 16. Smaller models are just the first P columns of the same draw. Nothing in this step is trained.

What is learned is a ridge regression on the last T months. The estimator β̂ is the matrix-inverse formula on the slide, and it is the solution to the minimisation on the right: average squared error over the window, plus z times the squared norm of β.

This is nonlinear in the signals G_t but linear in β̂. The lecture's way of saying it: a two-layer neural network with a frozen random first layer and a learned linear second layer.

One point to remember: the loss is squared error. The model does not optimise Sharpe. Sharpe is only how we evaluate it.

## Slide 6 — Strategy and metrics (5:00–6:15)

Here is how a forecast becomes a trade. At date t we fit β̂ on the previous T months, all with returns already realised. The forecast μ̂_t is S_t transpose β̂. The position π_t equals that forecast. The strategy earns π_t times next month's factor return. A positive forecast means long, a negative one means short, so this is timing, not static exposure. There is no look-ahead.

We report out-of-sample R squared, the coefficient norm, expected return, volatility and annualised Sharpe. Sharpe is computed per feature draw and per factor, averaged over draws, then over the 159 factors. That is an average of individual factor-timing Sharpe ratios, not the Sharpe of a 159-factor portfolio.

One useful fact: if you multiply β̂ by a positive constant, R squared and the coefficient norm change, but Sharpe does not. We'll see why that matters later.

## Slide 7 — Why P ≫ T can work (6:15–7:00)

Why doesn't this blow up? At c equal to one with almost no ridge, the model interpolates the sample. The coefficient norm and volatility spike, R squared crashes. That is the double-descent peak from the lecture.

Past c equal to one, the ridge solution is a minimum-norm fit that gets smoother with more features, and the norm falls. As c goes to infinity the random-feature model approaches kernel ridge regression. So "more complex" really means "a better-behaved smoother of the same signals", not "more overfit".

The paper's prediction at T equal to twelve follows. With small z, a dip near c equal to one and then recovery. With large z, Sharpe increasing in c. And the lecture's caveat applies: this does not license adding junk predictors.

**[Handoff]** "Yueqi will show what happened when we ran it."

## Slide 8 — Our implementation (7:00–8:00) — **Yueqi Lin**

We wrote this from the paper's equations; there is no public code that we used. The ridge is solved in its dual, T by T, form, so P equal to 7,000 is cheap. We use the same 159 factors and the same two signal sets, A and B. At T equal to twelve we use 100 feature draws, as in the paper. At sixty and one hundred twenty, ten draws.

What we cut: P is capped at 7,000 rather than 12,000, and we did not replicate futures or bonds, because that data is paid. Beyond the paper we added baselines, a placebo, an exact-kernel check and a simulator.

One ambiguity we hit: equations 36 and 37 in the paper imply different kernels. We implemented 37 literally and checked 36 as a sensitivity.

## Slide 9 — Figures (8:00–9:15)

On top is the paper's Figure 3, panels A and B, for T equal to twelve. Panel A is Welch–Goyal, B is momentum. Notice the broken axis: the far right is c near a thousand, so horizontal distance is not uniform. On the bottom are our versions, on a log axis. Each line is a different z.

Three things match. With large z, Sharpe rises steeply and then flattens. With small z, there is a dip right at c equal to one. And the curves for large z lie on top of each other. That last one is the scale invariance from earlier: at large z only the direction of β̂ matters.

## Slide 10 — Replication numbers (9:15–10:15)

In numbers, at T equal to twelve and large z. Welch–Goyal goes from 0.117 to 0.368, a factor of 3.1; the paper's curve goes from about 0.11 to 0.36. Momentum goes from 0.166 to 0.342, a factor of 2.1; the paper's from about 0.1 to 0.3. Paper values are read off the figure by eye. Ninety-six to ninety-eight percent of factors improve.

At longer windows the gain shrinks: about seven hundredths at sixty months and two to three hundredths at one hundred twenty. That is the same direction as the paper's Figure 5. At one hundred twenty months with tiny ridge, Sharpe actually falls.

What did not match: the coefficient norm plateaus in our version but falls in the paper, and the paper's ridge axis sits about three decades higher than ours. We suspect feature-normalisation conventions that the text leaves open.

**[Handoff]** "Hao will test how much of this is real."

## Slide 11 — Is the baseline fair? (10:15–11:15) — **Hao Li**

Now a question the paper does not ask. Its panels compare small and large models within the same family. They do not include an always-long benchmark.

So we added it. At T equal to twelve the model gets 0.342. Linear ridge on the same signals gets 0.11 to 0.16. The trailing twelve-month mean gets 0.338. And simply staying long the volatility-scaled factor gets 0.353.

The model clearly beats linear, matches the trailing mean, and ties always-long: minus 0.011 with a standard error of 0.017. One caveat on fairness: always-long is flattering, because the factors are signed using what the literature found worked. A real-time investor wouldn't know the signs.

## Slide 12 — Placebo (11:15–12:15)

Second test. We replaced the real signals with time-shuffled versions, which destroys any predictability, and reran everything. Orange is the placebo band; blue is the real signals.

At large z the rise is identical: plus 0.177 for the placebo against plus 0.176 for real signals. Real signals add roughly 0.02 on top, about four placebo standard deviations. That is detectable but small. At longer windows the real-signal edge is larger, about 0.05 to 0.07.

Our reading: most of the upward slope is a shrinkage effect. As c grows the random features converge to the kernel and the position becomes a smoother, lower-variance bet on the factor. It is not mainly learned timing.

## Slide 13 — Critique (12:15–13:15)

Where is the paper vulnerable?

One, most of the headline curve is shrinkage, as the placebo shows. Two, the comparison is across complexity levels, not against the obvious benchmark. Three, the factor list and signs use hindsight. Four, there are no transaction costs, on a nonlinear position that is refit monthly. Five, the specification is ambiguous in places, and conventions like the kernel scaling move the ridge axis. Six, our data vintage is newer than the paper's, so small differences are expected.

## Slide 14 — Takeaways (13:15–13:45)

To wrap up. Replicated: at T equal to twelve on 159 equity factors, Sharpe rises with complexity by two to three times, with the double-descent dip at small ridge. Qualified: the gain fades at longer windows and needs large ridge. Not shown: that complexity beats simply holding the factor, or that the rise is learned timing.

The lesson from implementing it: the shape of the curve is robust, but its economic value is not established. Code, data and every number are in our repository. We're happy to take questions.

---

## Likely questions, with the best person to answer

- **Why does Sharpe not depend on z at large z?** (Hongjun) Sharpe is invariant to a positive rescaling of the position, so when z is large only the direction of β̂ matters. Backup B.
- **Is there look-ahead?** (Hongjun) Signals and return scaling use past data only, and the ridge is fit on a window ending at t. Factor signs and the factor list do use hindsight (slide 11).
- **Why not just MSE or R² as the criterion?** (Hongjun) R² is negative at large P in our results while Sharpe rises: squared error punishes forecast scale, Sharpe does not. Backup A and B.
- **Why the placebo?** (Hao) To separate "learned something" from "a smoother estimator has less noise". It matches the real curve, so the rise itself is not evidence of learning.
- **Why not bonds or futures?** (Yueqi) Out of scope: the paper's data are paid or not available. We cover US equity factors only.
- **What about the paper's z axis?** (Yueqi) Ours sits about three decades lower. We suspect feature-normalisation conventions the paper does not fully specify.
- **Why is P in the paper called L?** (Luwei) Notation. The paper's L is the number of features, which the lecture calls P. We use the lecture's.

## Backup slides

**A. R² and ‖β̂‖ (T = 12).** R² at tiny z is −1.67 at c = 1 and −0.18 at c = 500. The norm peaks near c = 2, then falls, but plateaus where the paper's keeps falling.

**B. Squared error vs Sharpe.** Mean squared error is E[R²] − 2E[μ̂R] + E[μ̂²]: realised-return variation, a timing term, and forecast magnitude. A forecast with the right direction but wrong scale has poor R² and a good Sharpe, and scaling μ̂ by a positive constant leaves the Sharpe of π_t R_{t+1} unchanged.
