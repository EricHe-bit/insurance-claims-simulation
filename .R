# Insurance Claims Simulation in R

set.seed(42)

# Simulate annual losses
simulate_annual_losses <- function(n_years, lambda, severity_dist, params) {
  counts <- rpois(n_years, lambda)
  totals <- numeric(n_years)
  
  for (i in 1:n_years) {
    c <- counts[i]
    if (c > 0) {
      if (severity_dist == "exponential") {
        sev <- rexp(c, rate = 1/params$mean)
      } else if (severity_dist == "normal") {
        sev <- pmax(rnorm(c, mean = params$mean, sd = params$sd), 0) # truncate at 0
      } else if (severity_dist == "lognormal") {
        sev <- rlnorm(c, meanlog = params$mu, sdlog = params$sigma)
      }
      totals[i] <- sum(sev)
    }
  }
  return(totals)
}

# Stop-loss function
stop_loss <- function(total_losses, attachment, limit = Inf) {
  excess <- pmax(total_losses - attachment, 0)
  ceded <- pmin(excess, limit)
  net <- total_losses - ceded
  return(list(ceded = ceded, net = net))
}

# Parameters
n_years <- 50000
lambda <- 10

exp_params <- list(mean = 5000)
norm_params <- list(mean = 5000, sd = 2000)
sigma <- 0.9
mu <- log(7000) - 0.5*(sigma^2)
lognorm_params <- list(mu = mu, sigma = sigma)

# Run simulations
sev_types <- list(
  exponential = exp_params,
  normal = norm_params,
  lognormal = lognorm_params
)

results <- list()

for (name in names(sev_types)) {
  gross <- simulate_annual_losses(n_years, lambda, name, sev_types[[name]])
  sl <- stop_loss(gross, 80000, 100000)
  results[[name]] <- list(gross = gross, ceded = sl$ceded, net = sl$net)
}

# Metrics function
metrics <- function(x) {
  c(mean = mean(x), variance = var(x), VaR95 = quantile(x, 0.95))
}

# Display results in a nice table format
cat("Insurance Claims Simulation - Metrics\n")
cat("==================================================\n\n")

for (name in names(results)) {
  display_name <- switch(name,
                        "exponential" = "Exponential",
                        "normal" = "Normal (truncated ≥0)",
                        "lognormal" = "Lognormal")
  
  gross_metrics <- metrics(results[[name]]$gross)
  net_metrics <- metrics(results[[name]]$net)
  ceded_metrics <- metrics(results[[name]]$ceded)
  
  cat(paste0("Severity: ", display_name, "\n"))
  cat(sprintf("Gross - Mean: $%.2f, Variance: %.2f, VaR 95%%: $%.2f\n", 
              gross_metrics["mean"], gross_metrics["variance"], gross_metrics["VaR95%"]))
  cat(sprintf("Net   - Mean: $%.2f, Variance: %.2f, VaR 95%%: $%.2f\n", 
              net_metrics["mean"], net_metrics["variance"], net_metrics["VaR95%"]))
  cat(sprintf("Ceded - Mean: $%.2f, Variance: %.2f, VaR 95%%: $%.2f\n\n", 
              ceded_metrics["mean"], ceded_metrics["variance"], ceded_metrics["VaR95%"]))
}

# --- Visualizations ---
library(ggplot2)
library(gridExtra)

# 1) Severity *shape* examples (draw 100k severities from each to visualize distribution)
set.seed(42)
sev_draws <- list(
  Exponential = rexp(100000, rate = 1/5000),
  "Normal (truncated ≥0)" = pmax(rnorm(100000, mean = 5000, sd = 2000), 0),
  Lognormal = rlnorm(100000, meanlog = mu, sdlog = sigma)
)

# Create severity distribution plots
severity_plots <- list()
for (name in names(sev_draws)) {
  df <- data.frame(claim_size = sev_draws[[name]])
  p <- ggplot(df, aes(x = claim_size)) +
    geom_histogram(bins = 100, fill = "steelblue", alpha = 0.7, color = "black") +
    labs(title = paste("Severity Distribution:", name),
         x = "Claim Size ($)",
         y = "Frequency") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  severity_plots[[name]] <- p
}

# Display severity plots
grid.arrange(grobs = severity_plots, ncol = 1)

# 2) Annual aggregate losses (gross) histograms
gross_plots <- list()
for (name in names(results)) {
  display_name <- switch(name,
                        "exponential" = "Exponential",
                        "normal" = "Normal (truncated ≥0)",
                        "lognormal" = "Lognormal")
  
  df <- data.frame(annual_loss = results[[name]]$gross)
  p <- ggplot(df, aes(x = annual_loss)) +
    geom_histogram(bins = 100, fill = "darkgreen", alpha = 0.7, color = "black") +
    labs(title = paste("Annual Aggregate Losses (Gross) —", display_name),
         x = "Annual Total Loss ($)",
         y = "Frequency") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  gross_plots[[name]] <- p
}

# Display gross loss plots
grid.arrange(grobs = gross_plots, ncol = 1)

# 3) Risk transfer effect for Lognormal: Gross vs Net
log_gross <- results$lognormal$gross
log_net <- results$lognormal$net

df_risk <- data.frame(
  Loss = c(log_gross, log_net),
  Type = rep(c("Gross", "Net (Stop-Loss)"), each = length(log_gross))
)

risk_transfer_plot <- ggplot(df_risk, aes(x = Loss, fill = Type)) +
  geom_histogram(bins = 100, alpha = 0.6, position = "identity", color = "black") +
  labs(title = "Risk Transfer: Gross vs Net — Lognormal Severity",
       x = "Annual Total Loss ($)",
       y = "Frequency",
       fill = "Loss Type") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = "top")

print(risk_transfer_plot)

# --- Save results ---
# Create a comprehensive results dataframe
results_df <- data.frame()
for (name in names(results)) {
  display_name <- switch(name,
                        "exponential" = "Exponential",
                        "normal" = "Normal (truncated ≥0)",
                        "lognormal" = "Lognormal")
  
  gross_metrics <- metrics(results[[name]]$gross)
  net_metrics <- metrics(results[[name]]$net)
  ceded_metrics <- metrics(results[[name]]$ceded)
  
  results_df <- rbind(results_df,
                     data.frame(Severity = display_name,
                                Type = "Gross",
                                Mean = gross_metrics["mean"],
                                Variance = gross_metrics["variance"],
                                VaR_95 = gross_metrics["VaR95%"]),
                     data.frame(Severity = display_name,
                                Type = "Net (Stop-Loss 80k xs 100k)",
                                Mean = net_metrics["mean"],
                                Variance = net_metrics["variance"],
                                VaR_95 = net_metrics["VaR95%"]),
                     data.frame(Severity = display_name,
                                Type = "Ceded",
                                Mean = ceded_metrics["mean"],
                                Variance = ceded_metrics["variance"],
                                VaR_95 = ceded_metrics["VaR95%"]))
}

# Save to CSV
write.csv(results_df, "insurance_claims_simulation_metrics.csv", row.names = FALSE)
cat("✓ Metrics saved to: insurance_claims_simulation_metrics.csv\n")

cat("\nSimulation completed successfully!\n")