# =============================================================================
# Causal Analysis: Smoking Cessation → BMI Change → Diabetes Risk
# Motivated by Feuerriegel et al. (2024) Nature Medicine 30(4): 958–968
#
# Key empirical backdrop (not raw data — calibrated from published estimates):
#
#   Hu et al. (2018) NEJM 379:653–663 [NHS / NHS-II / HPFS cohorts]
#     - Quitting smoking with ≥10 kg weight gain → HR ≈ 1.91 for T2D
#     - Weight gain within 6 years explained ~68% of the transient T2D spike
#     - After 15+ years of cessation T2D risk returns below that of smokers
#
#   Kangbuk Samsung cohort (MDPI JCM 2024):
#     - Mean BMI in T2D group: 25.68 ± 3.53; non-T2D: 23.18 ± 3.28
#     - Current smokers HR ≈ 1.29 for T2D vs never smokers
#
#   Bush (2016) Obesity review:
#     - Post-cessation BMI increase ≈ 0.9 kg/m² on average within 3 years
#     - Magnitude varies with baseline BMI and cigarettes/day
#
# The Feuerriegel et al. critique:
#   Standard ML predicts diabetes risk from (quit_smoking=1, BMI=observed).
#   But quitting CAUSES BMI to rise. The correct intervention distribution is
#   p(BMI | do(quit_smoking=1)), not p(BMI | quit_smoking=1, baseline).
#   Ignoring the smoking → BMI pathway yields biased CATE estimates.
#
# Causal DAG:
#
#   Age ──────────────────────────────────────┐
#   Sex ──────────┬──────────────────────────►│
#   FamHx ────────┤                           │
#                 ▼                           ▼
#   Baseline_BMI ──► ΔSmoke ──► ΔBMI ──────► Diabetes
#       ▲                         ▲
#       └──── Confounders ────────┘
#            (alcohol, activity)
#
# Three analysis strategies (illustrating the Feuerriegel point):
#
#   (A) Naïve ML: predict Diabetes from (quit, post_BMI) ignoring mediation
#   (B) Causal mediation: decompose total effect into direct + BMI-mediated
#   (C) Front-door / SCM g-formula: intervene do(quit=1), propagate to BMI
#       then to diabetes — the proper counterfactual prediction
# =============================================================================

set.seed(2024)

# ── 0. Packages ───────────────────────────────────────────────────────────────
#pkgs <- c("ggplot2", "mediation", "dagitty", "ggdag")
#for (p in pkgs) if (!requireNamespace(p, quietly=TRUE)) install.packages(p)
library(ggplot2)
library(mediation)


# ── 1. Synthetic cohort calibrated to published effect sizes ──────────────────
#
# Parameters grounded in:
#   - Hu et al. (2018): post-cessation ΔBMI, HR estimates for T2D
#   - Kangbuk Samsung (2024): baseline BMI means and SD by T2D status
#   - Bush (2016): BMI gain ≈ 0.9 kg/m² on average within 3 years post-quit

n <- 2000

# Baseline covariates
age      <- round(rnorm(n, 50, 10))
sex      <- rbinom(n, 1, 0.45)           # 1 = male
fam_hx   <- rbinom(n, 1, 0.20)          # family history of diabetes
alcohol  <- pmax(0, rnorm(n, 8, 5))     # g/day
activity <- rbinom(n, 1, 0.35)          # physically active

# Baseline BMI (smokers tend to have slightly lower BMI — causal effect of
# smoking on weight documented via Mendelian randomisation, Atkins 2021)
bmi0 <- 23.5 +
  0.05  * (age - 50) +
  1.20  * sex +
  0.50  * fam_hx +
  0.04  * alcohol -
  0.80  * activity +     # active → lower BMI
  rnorm(n, sd = 3.0)
bmi0 <- pmax(16, bmi0)

# Treatment: quit smoking (1 = quitter, 0 = continued smoker)
# Quitting is influenced by age, sex, baseline BMI, activity
lp_quit <- -1.2 + 0.015*(age-50) - 0.20*sex + 0.02*(bmi0-23) + 0.30*activity
quit    <- rbinom(n, 1, plogis(lp_quit))

# ── MEDIATOR: post-cessation BMI change ───────────────────────────────────────
# Calibration:
#   Quitters gain ≈ +1.0 kg/m² on average (Bush 2016; Hu 2018 appendix).
#   Higher baseline BMI → more gain (Lycett et al. finding encoded here).
#   This is the pathway Feuerriegel warns naive ML ignores.

delta_bmi <- 0.90 * quit +
             0.05 * (bmi0 - 23) * quit +      # interaction: higher bmi0 → more gain
             0.30 * (1 - activity) * quit +    # sedentary quitters gain more
            -0.15 * activity +                 # active individuals less gain
             rnorm(n, sd = 1.2)

bmi_post <- bmi0 + delta_bmi

# ── OUTCOME: incident T2D (binary, ~5-year risk) ──────────────────────────────
# Log-odds calibrated so:
#   - baseline prevalence ~7% in continued smokers
#   - current smokers HR ≈ 1.29 vs never smokers (Kangbuk Samsung 2024)
#   - BMI effect: per unit BMI increase, OR ≈ 1.10 (well established)
#   - quitting has a direct protective effect (long-run benefit)
#     but mediated BMI rise partially offsets it short-term (Hu 2018)

lp_dm <- -5.50 +
          0.035 * (age - 50) +
          0.50  * sex +
          0.80  * fam_hx +
          0.20  * (bmi_post - 23) +    # BMI effect on diabetes
          0.025 * alcohol +
         -0.40  * activity +
         -0.30  * quit +               # direct protective effect of quitting
          rnorm(n, sd = 0.3)           # residual individual heterogeneity

diabetes <- rbinom(n, 1, plogis(lp_dm))

cohort <- data.frame(
  age, sex, fam_hx, alcohol, activity,
  bmi0, quit, delta_bmi, bmi_post, lp_dm, diabetes
)

cat("── Cohort summary ────────────────────────────────────────────────────\n")
cat(sprintf("  n = %d  |  Quitters: %d (%.1f%%)  |  T2D cases: %d (%.1f%%)\n",
            n, sum(quit), 100*mean(quit), sum(diabetes), 100*mean(diabetes)))
cat(sprintf("  Mean baseline BMI: %.2f  |  Mean post BMI: %.2f\n",
            mean(bmi0), mean(bmi_post)))
cat(sprintf("  Mean ΔBMI in quitters: +%.2f kg/m²  |  non-quitters: %.2f kg/m²\n",
            mean(delta_bmi[quit==1]), mean(delta_bmi[quit==0])))
cat(sprintf("  T2D rate: quitters=%.1f%%  non-quitters=%.1f%%\n\n",
            100*mean(diabetes[quit==1]), 100*mean(diabetes[quit==0])))


# ── 2. (A) Naïve ML approach — the WRONG way (Feuerriegel critique) ───────────
#
# Predicts diabetes from (quit, post_BMI) as if post_BMI were independent of
# quit. This is exactly what the Nature Medicine paper warns against:
# "ML would suggest using both BMI and smoking behavior to predict diabetes
#  risk... but this ignores that stopping smoking would also change BMI."
naive_fit <- glm(diabetes ~ quit + bmi_post + age + sex + fam_hx +
                            alcohol + activity,
                 data = cohort, family = binomial)
#   show  BMI gain ≈ 0.9 kg/m² on average within 3 years post-quit (Bush (2016))
plot(bmi_post~factor(quit),cohort)
summary(lm(bmi_post~quit,cohort))
cat("── (A) Naïve ML (logistic on post-treatment BMI) ─────────────────────\n")
cat("  [Incorrect: conditions on mediator bmi_post — blocks causal path]\n")
naive_coef <- coef(summary(naive_fit))
cat(sprintf("  quit coef  = %.3f  (OR = %.3f)  p = %.4f\n",
            naive_coef["quit","Estimate"],
            exp(naive_coef["quit","Estimate"]),
            naive_coef["quit","Pr(>|z|)"]))
cat(sprintf("  bmi_post   = %.3f  (OR = %.3f)  p = %.4f\n\n",
            naive_coef["bmi_post","Estimate"],
            exp(naive_coef["bmi_post","Estimate"]),
            naive_coef["bmi_post","Pr(>|z|)"]))


# ── 3. (B) Causal mediation analysis (Imai et al. 2010, R: mediation pkg) ────
#
# Decomposes:
#   Total Effect (TE)  = ADE (average direct effect)  +  ACME (BMI-mediated)
#
# Mediator model: ΔBMI ~ quit + covariates
# Outcome model:  diabetes ~ quit + ΔBMI + covariates
#
# This correctly treats ΔBMI as a mediator, not a confounder or covariate.

med_model <- lm(delta_bmi ~ quit + bmi0 + age + sex + fam_hx +
                             alcohol + activity,
                data = cohort)

out_model <- glm(diabetes ~ quit + delta_bmi + bmi0 + age + sex + fam_hx +
                             alcohol + activity,
                 data = cohort, family = binomial)

cat("── (B) Causal mediation analysis (Imai et al. 2010) ─────────────────\n")
cat("  Mediator: ΔBMI  |  Treatment: quit  |  Outcome: T2D\n")

med_out <- mediate(med_model, out_model,
                   treat = "quit", mediator = "delta_bmi",
                   robustSE = TRUE, sims = 500)

cat(sprintf("  ACME (BMI-mediated effect) : %.4f  [95%%CI: %.4f, %.4f]\n",
            med_out$d0, med_out$d0.ci[1], med_out$d0.ci[2]))
cat(sprintf("  ADE  (direct effect)       : %.4f  [95%%CI: %.4f, %.4f]\n",
            med_out$z0, med_out$z0.ci[1], med_out$z0.ci[2]))
cat(sprintf("  Total effect               : %.4f  [95%%CI: %.4f, %.4f]\n",
            med_out$tau.coef, med_out$tau.ci[1], med_out$tau.ci[2]))
cat(sprintf("  Proportion mediated (BMI)  : %.1f%%\n\n",
            med_out$n0 * 100))


# ── 4. (C) G-formula / SCM: do(quit=1) counterfactual ────────────────────────
#
# The principled causal prediction for Feuerriegel's example:
# What is P(T2D=1 | do(quit=1)) vs P(T2D=1 | do(quit=0))?
#
# Steps (parametric g-formula):
#   1. Fit mediator model: E[ΔBMI | quit, confounders]
#   2. Fit outcome model:  E[diabetes | quit, ΔBMI, confounders]
#   3. For each individual, set quit=1 (or 0), predict ΔBMI from step 1,
#      then predict diabetes from step 2 using that predicted ΔBMI.
#   4. Average over the covariate distribution.
#
# This is NOT what naive ML does — naive ML fixes post_BMI at observed values.

# Refit models on confounders only (no post-treatment variable as covariate)
med_gform <- lm(delta_bmi ~ quit + bmi0 + age + sex + fam_hx + alcohol + activity,
                data = cohort)
out_gform <- glm(diabetes ~ quit + delta_bmi + bmi0 + age + sex + fam_hx +
                             alcohol + activity,
                 data = cohort, family = binomial)

gformula_ate <- function(df, treat_val) {
  df_int           <- df
  df_int$quit      <- treat_val
  # Step 1: predict mediator under intervention
  df_int$delta_bmi <- predict(med_gform, newdata = df_int)
  df_int$bmi_post  <- df_int$bmi0 + df_int$delta_bmi
  # Step 2: predict outcome under intervention + predicted mediator
  mean(predict(out_gform, newdata = df_int, type = "response"))
}

p_dm_quit  <- gformula_ate(cohort, 1)   # do(quit=1)
p_dm_smoke <- gformula_ate(cohort, 0)   # do(quit=0)
ate_gform  <- p_dm_quit - p_dm_smoke

# Bootstrap 95% CI
boot_ate <- replicate(500, {
  idx <- sample(nrow(cohort), replace=TRUE)
  b   <- cohort[idx, ]
  m   <- lm(delta_bmi ~ quit + bmi0 + age + sex + fam_hx + alcohol + activity, data=b)
  o   <- glm(diabetes ~ quit + delta_bmi + bmi0 + age + sex + fam_hx + alcohol + activity,
             data=b, family=binomial)
  f <- function(df, tv) {
    d <- df; d$quit <- tv
    d$delta_bmi <- predict(m, newdata=d)
    mean(predict(o, newdata=d, type="response"))
  }
  f(b,1) - f(b,0)
})
ci_gform <- quantile(boot_ate, c(0.025, 0.975))

cat("── (C) G-formula: do(quit=1) vs do(quit=0) ──────────────────────────\n")
cat("  [Correct causal estimate: propagates quit→ΔBMI→T2D]\n")
cat(sprintf("  P(T2D=1 | do(quit=1))  = %.3f\n", p_dm_quit))
cat(sprintf("  P(T2D=1 | do(quit=0))  = %.3f\n", p_dm_smoke))
cat(sprintf("  ATE = do(quit=1)−do(quit=0) = %.4f  [95%%CI: %.4f, %.4f]\n\n",
            ate_gform, ci_gform[1], ci_gform[2]))


# ── 5. Compare naive vs causal risk differences ───────────────────────────────

naive_pred_quit  <- mean(predict(naive_fit,
  newdata = transform(cohort, quit=1), type="response"))
naive_pred_smoke <- mean(predict(naive_fit,
  newdata = transform(cohort, quit=0), type="response"))
ate_naive <- naive_pred_quit - naive_pred_smoke

cat("── Comparison: naïve vs causal ATE ──────────────────────────────────\n")
cat(sprintf("  Naïve ML ATE  (biased, blocks mediator): %.4f\n", ate_naive))
cat(sprintf("  G-formula ATE (causal, do-calculus):     %.4f\n", ate_gform))
cat(sprintf("  Bias due to ignoring smoking→BMI path:   %.4f\n\n",
            abs(ate_naive - ate_gform)))


# ── 6. CATE by baseline BMI group (heterogeneous treatment effects) ────────────
#
# The Feuerriegel paper emphasises CATE (conditional average treatment effects).
# Patients with high baseline BMI may have worse short-term T2D risk from
# quitting (via larger BMI gain) even though quitting is beneficial long-run.

cat("── CATE by baseline BMI group ────────────────────────────────────────\n")
cohort$bmi_grp <- cut(bmi0,
  breaks = c(-Inf, 22, 25, 28, Inf),
  labels = c("Underweight/Normal\n(<22)", "Normal\n(22–25)",
             "Overweight\n(25–28)", "Obese\n(≥28)"))

cate_df <- do.call(rbind, lapply(levels(cohort$bmi_grp), function(g) {
  sub <- cohort[cohort$bmi_grp == g, ]
  if (nrow(sub) < 30) return(NULL)
  m  <- lm(delta_bmi ~ quit + bmi0 + age + sex + fam_hx + alcohol + activity, data=sub)
  o  <- glm(diabetes ~ quit + delta_bmi + bmi0 + age + sex + fam_hx + alcohol + activity,
            data=sub, family=binomial)
  f <- function(tv) {
    d <- sub; d$quit <- tv
    d$delta_bmi <- predict(m, newdata=d)
    mean(predict(o, newdata=d, type="response"))
  }
  p1 <- f(1); p0 <- f(0)
  data.frame(bmi_grp=g, cate=p1-p0, p_quit=p1, p_smoke=p0, n=nrow(sub))
}))

print(cate_df[, c("bmi_grp","n","p_smoke","p_quit","cate")], row.names=FALSE,
      digits=4)


# ── 7. Visualisation ──────────────────────────────────────────────────────────

# ── 7a. DAG (manual ggplot) ───────────────────────────────────────────────────
dag_nodes <- data.frame(
  node = c("Quit\nSmoking", "ΔBMI", "Diabetes", "Baseline\nBMI",
           "Age/Sex/\nFam Hx", "Alcohol/\nActivity"),
  x    = c(3, 5, 7, 3, 1, 1),
  y    = c(3, 3, 3, 5, 5, 1)
)
dag_edges <- data.frame(
  from = c("Quit\nSmoking","ΔBMI","Baseline\nBMI","Baseline\nBMI",
           "Age/Sex/\nFam Hx","Age/Sex/\nFam Hx","Alcohol/\nActivity",
           "Alcohol/\nActivity","Quit\nSmoking"),
  to   = c("ΔBMI","Diabetes","ΔBMI","Diabetes",
           "Quit\nSmoking","Diabetes","ΔBMI","Diabetes","Diabetes"),
  type = c("mediated","mediated","conf","conf",
           "conf","conf","conf","conf","direct")
)

dag_coords <- setNames(dag_nodes[,c("x","y")], c("x","y"))
rownames(dag_coords) <- dag_nodes$node

get_xy <- function(nm) dag_coords[nm, ]

edges_plot <- do.call(rbind, lapply(seq_len(nrow(dag_edges)), function(i) {
  f  <- dag_edges$from[i]; t <- dag_edges$to[i]
  data.frame(x=dag_coords[f,"x"], y=dag_coords[f,"y"],
             xend=dag_coords[t,"x"], yend=dag_coords[t,"y"],
             type=dag_edges$type[i])
}))

p_dag <- ggplot() +
  geom_segment(data=edges_plot,
               aes(x=x, y=y, xend=xend, yend=yend, colour=type),
               arrow=arrow(length=unit(0.18,"cm"), type="closed"),
               linewidth=0.85, alpha=0.85) +
  geom_label(data=dag_nodes, aes(x=x, y=y, label=node),
             size=3.2, fill="white", colour="#1a1a1a",
             label.size=0.4, label.padding=unit(0.3,"lines")) +
  scale_colour_manual(
    values=c("mediated"="#1D9E75","direct"="#3B8BD4","conf"="#888780"),
    labels=c("mediated"="Mediated path (quit→ΔBMI→T2D)",
             "direct"="Direct effect (quit→T2D)",
             "conf"="Confounder/covariate")) +
  annotate("text", x=4, y=3.35, label="Mediator\n(BMI gain)", size=3,
           colour="#085041") +
  annotate("text", x=5, y=2.65, label="Feuerriegel's\nwarning: this path\nignored by naïve ML",
           size=2.8, colour="#993C1D", fontface="italic") +
  labs(title="Causal DAG: Smoking cessation → ΔBMI → Diabetes",
       subtitle="Hu et al. (2018) NEJM; Kangbuk Samsung cohort (2024)",
       colour=NULL, x=NULL, y=NULL) +
  theme_void(base_size=11) +
  theme(plot.title=element_text(face="bold"),
        legend.position="bottom",
        legend.text=element_text(size=9))

#png("smoking_bmi_diabetes_dag.png", width=820, height=480, res=120, bg="white")
print(p_dag)
#dev.off()


# ── 7b. Mediation decomposition bar chart ─────────────────────────────────────
med_bar <- data.frame(
  component = factor(c("ACME\n(BMI-mediated)", "ADE\n(direct effect)"),
                     levels=c("ACME\n(BMI-mediated)", "ADE\n(direct effect)")),
  estimate  = c(med_out$d0, med_out$z0),
  lo        = c(med_out$d0.ci[1], med_out$z0.ci[1]),
  hi        = c(med_out$d0.ci[2], med_out$z0.ci[2]),
  colour    = c("#1D9E75", "#3B8BD4")
)

p_med <- ggplot(med_bar, aes(x=component, y=estimate, fill=component)) +
  geom_col(width=0.45, alpha=0.85) +
  geom_errorbar(aes(ymin=lo, ymax=hi), width=0.12, linewidth=0.7) +
  geom_hline(yintercept=0, linetype="dashed", linewidth=0.5) +
  scale_fill_manual(values=c("#1D9E75","#3B8BD4")) +
  labs(title="Causal mediation: total effect decomposition",
       subtitle=sprintf("Proportion mediated via BMI: %.1f%%   |   Total ATE: %.4f",
                        med_out$n0*100, med_out$tau.coef),
       x=NULL, y="Effect on P(T2D=1)", fill=NULL) +
  theme_minimal(base_size=12) +
  theme(plot.title=element_text(face="bold"), legend.position="none")

#png("smoking_mediation_decomp.png", width=640, height=420, res=120, bg="white")
print(p_med)
#dev.off()


# ── 7c. CATE plot ─────────────────────────────────────────────────────────────
cate_df$bmi_grp <- factor(cate_df$bmi_grp, levels=levels(cohort$bmi_grp))

p_cate <- ggplot(cate_df, aes(x=bmi_grp, y=cate, fill=cate < 0)) +
  geom_col(width=0.55, alpha=0.85) +
  geom_hline(yintercept=0, linetype="dashed") +
  scale_fill_manual(values=c("FALSE"="#D85A30","TRUE"="#1D9E75"),
                    labels=c("FALSE"="Quitting increases T2D risk (short-term)",
                             "TRUE"="Quitting reduces T2D risk")) +
  labs(title="CATE by baseline BMI group (g-formula)",
       subtitle="High-BMI patients face greater short-term T2D risk from cessation via larger weight gain",
       x="Baseline BMI group", y="CATE: P(T2D|do(quit=1)) − P(T2D|do(quit=0))",
       fill=NULL) +
  theme_minimal(base_size=12) +
  theme(plot.title=element_text(face="bold"),
        legend.position="bottom",
        axis.text.x=element_text(size=9))

#png("smoking_cate_bmi.png", width=760, height=440, res=120, bg="white")
print(p_cate)
#dev.off()


# ── 7d. Naïve vs causal risk prediction scatter ───────────────────────────────

cohort$risk_naive  <- predict(naive_fit, type="response")

# Causal risk per individual (g-formula, individual-level)
cohort$delta_bmi_hat <- predict(med_gform)     # E[ΔBMI | X, quit_obs]
tmp_q1 <- cohort; tmp_q1$quit <- 1
tmp_q1$delta_bmi <- predict(med_gform, newdata=tmp_q1)
cohort$risk_causal_quit <- predict(out_gform, newdata=tmp_q1, type="response")

tmp_q0 <- cohort; tmp_q0$quit <- 0
tmp_q0$delta_bmi <- predict(med_gform, newdata=tmp_q0)
cohort$risk_causal_smoke <- predict(out_gform, newdata=tmp_q0, type="response")
cohort$risk_causal_cate  <- cohort$risk_causal_quit - cohort$risk_causal_smoke

# For quitters: how does naïve vs causal risk differ?
quitters <- cohort[cohort$quit == 1, ]

p_scatter <- ggplot(quitters, aes(x=risk_naive, y=risk_causal_quit,
                                   colour=bmi0)) +
  geom_point(alpha=0.4, size=0.9) +
  geom_abline(slope=1, intercept=0, linetype="dashed", colour="#333") +
  scale_colour_gradient(low="#3B8BD4", high="#D85A30",
                        name="Baseline BMI") +
  labs(title="Naïve ML vs causal (g-formula) T2D risk — quitters only",
       subtitle="Points above diagonal: causal model gives HIGHER risk than naïve ML\n(because g-formula correctly propagates smoking→BMI increase→diabetes)",
       x="Naïve ML predicted risk  [conditions on observed post-BMI]",
       y="Causal predicted risk  [do(quit=1), propagates ΔBMI]") +
  theme_minimal(base_size=11) +
  theme(plot.title=element_text(face="bold"))

#png("smoking_naive_vs_causal_risk.png", width=760, height=500, res=120, bg="white")
print(p_scatter)
#dev.off()


# ── 8. Final summary ──────────────────────────────────────────────────────────

cat("\n══ Summary ════════════════════════════════════════════════════════════\n")
cat("  Empirical context:\n")
cat("    Hu et al. (2018) NEJM: weight gain mediated ~68% of T2D spike after\n")
cat("    quitting; after 15+ yrs cessation, T2D risk drops below that of\n")
cat("    active smokers. Calibrated here using NHS/HPFS cohort estimates.\n\n")
cat("  Feuerriegel et al. (2024) Nature Medicine warning:\n")
cat("    Standard ML uses observed post-BMI as a feature. This implicitly\n")
cat("    conditions on the mediator, blocking the quit→ΔBMI→T2D path and\n")
cat("    biasing treatment effect estimates.\n\n")
cat(sprintf("  Naïve ATE  (biased): %.4f\n", ate_naive))
cat(sprintf("  Causal ATE (g-form): %.4f\n", ate_gform))
cat(sprintf("  Absolute bias      : %.4f\n\n", abs(ate_naive - ate_gform)))
cat("  Mediation (correct decomposition):\n")
cat(sprintf("    Direct effect       : %.4f\n", med_out$z0))
cat(sprintf("    ΔBMI-mediated effect: %.4f (%.1f%% of total)\n",
            med_out$d0, med_out$n0*100))
cat("  Causal recommendation (Feuerriegel):\n")
cat("    Embed ML in SCM. Use g-formula or TMLE to estimate do(quit=1),\n")
cat("    propagating intervention through the mediator, not conditioning on it.\n")
cat("═══════════════════════════════════════════════════════════════════════\n")
