library(nett)
library(hsbm)
library(ggplot2)
library(dplyr)
library(doParallel)

seed = 1400
ncores = min(parallel::detectCores() - 1, 32) # number of cores to use to parallel
nreps = 500 # number of replications
ntrans = 8
seq_g_update = F
save_data = T
n = 200 # number of nodes in each layer
nlayers = 5 # number of layers
random_net = T

niter <- 100 # number of Gibbs iterations 
burnin <- ceiling(niter/2)

tau = 0.0
methods = list()
methods[["HSBM"]] = function(A, K) {# K is not used
  zh = fit_hsbm(A, beta0=0.1, gam0=.5, niter=niter, Kcap=10, Gcap=10,
                seq_g_update = seq_g_update, verb = F)$zb
  get_map_labels(zh, burnin = burnin, consecutive = T)$labels
}
methods[["DP-SBM"]] =  function(A, K) {# K is not used
   zh = fit_mult_dpsbm(A, gam0=.5, niter=niter, Zcap=10, verb = F)$zb
   get_map_labels(zh, burnin = burnin, consecutive = T)$labels
}
methods[["SC-sliced"]] = function(A, K) spec_clust_sliced(A, K, tau = tau)
methods[["SC-avg"]] = function(A, K) spec_clust_avg(A, K, tau = tau)
methods[["SC-ba"]] = function(A, K) spec_clust_bias_adj(A, K)
methods[["SC-omni"]] = function(A, K) spec_clust_omnibus(A, K)
methods[["PisCES"]] = function(A, K) pisces(A, K, shared_kmeans_init = F, verb = F)
methods[["PisCES-sh"]] = function(A, K) pisces(A, K, shared_kmeans_init = T, verb = F)

mtd_names = names(methods)

runs = expand.grid(
  mtd_idx = seq_along(methods), 
  trans_prob = seq(0, 1, length.out = ntrans),
  rep = 1:nreps  
)

cl <- parallel::makeForkCluster(ncores)
doParallel::registerDoParallel(cl)
doRNG::registerDoRNG(seed)

gen_rand_eta = function() {
  eta = matrix(0,3,3)
  eta[upper.tri(eta, diag=T)] = runif(3*4/2, 0.1,0.9)
  (eta + t(eta))/2
}

total_time = system.time(
  #res <- do.call(rbind, parallel::mclapply(1:nrow(runs), function(j) {
  #res <-  do.call(rbind, lapply(1:nrow(runs), function(j) {
  res <- do.call(rbind, foreach::foreach(j = 1:nrow(runs)) %dopar% {
    mi = runs[j, "mtd_idx"]
    trans_prob = runs[j, "trans_prob"]
    if (random_net) {
      out = sample_markov_sbm(n, nlayers, gen_rand_eta(), trans_prob = trans_prob)
    } else {
      out = sample_personality_net(n, nlayers, trans_prob = trans_prob) # , seed=1400)   
    }
    Ktru = nrow(out$eta)
    A = out$A
    zb = out$zb
    
    dt = as.numeric(system.time( zh <- methods[[mi]](A, Ktru) )["elapsed"])
    data.frame(method = mtd_names[mi], 
               aggregate_nmi = get_agg_nmi(zb, zh), 
               slicewise_nmi = get_slice_nmi(zb, zh) , 
               elapsed_time = dt, trans_prob = trans_prob)
  })
  #}))
  #}, mc.cores = ncores, mc.cleaup = T))
)["elapsed"]
nett::printf("Total simulation time = %3.2f (s)\n" , total_time)

if (random_net) {
  file_tag = sprintf("markov_exp_rnd_n%d_nla%d_nre%d_ntr%d_seq%d", n, nlayers, nreps, ntrans, seq_g_update)
  yrange = c(0, 0.8)
} else {
  file_tag = sprintf("markov_exp_n%d_nla%d_nre%d_ntr%d_seq%d", n, nlayers, nreps, ntrans, seq_g_update)  
  yrange = c(0, 1)
}

if (save_data) save.image(paste0(file_tag,".RData"))


res %>% 
  group_by(method, trans_prob) %>% 
  summarise_all(mean) %>% 
  mutate(method = factor(method, levels = mtd_names)) %>%   # Make the ordered in the order they are defined
  ggplot(aes(x = trans_prob, y = aggregate_nmi, color = method, linetype = method, shape = method)) + 
  geom_line(size = 1.25) +
  # scale_y_continuous( trans = "log10") +
  geom_point(size = 3) +
  theme_minimal(base_size = 16) +
  xlab("Transition Probability") +
  ylab("Aggregate NMI") + 
  ylim(yrange) +
  # ylim(c(0,1)) +
  ggplot2::theme(
    legend.background = ggplot2::element_blank(),
    legend.title = ggplot2::element_blank(),
    legend.position = c(0.8, 0.5),
    # legend.text = ggplot2::element_text(size=18),
  ) + 
  ggplot2::guides(colour = ggplot2::guide_legend(keywidth = 4, keyheight = 1.25))

ggsave(paste0(file_tag, ".pdf"), width = 7)
  
sum_res = res %>% 
  filter(trans_prob == unique(res$trans_prob)[5]) %>% 
  select(-trans_prob) %>% 
  group_by(method) %>% 
  summarise(agg_mean = mean(aggregate_nmi), agg_sd = sd(aggregate_nmi), # sd(aggregate_nmi)/sqrt(nreps), 
            slice_mean = mean(slicewise_nmi), slice_sd = sd(slicewise_nmi), 
            time_mean = mean(elapsed_time), time_sd = sd(elapsed_time), ) %>% 
  arrange(desc(agg_mean))

print( knitr::kable(sum_res, digits = 2, format="pipe") )


  
 
