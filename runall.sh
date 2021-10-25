for d in celeba; #cocos3 cocos3_10k; # celeba celeba_private;
do
    for seed in 0 1 2;
    do
#         python3.6 "$d".py --explorer bern_gp --et random --seed $seed > run.log 2>&1 &
#         python "$d".py --explorer bern_gp2 --et svariance2 --seed $seed --width 1 > run.log 2>&1 &
#         python "$d".py --explorer bern_gp2 --et svariance2 --seed $seed --width 3 > run.log 2>&1 &
#         python "$d".py --explorer beta_gp_rloss --et svariance2 --seed $seed --width 1 > run.log 2>&1 &
#         python "$d".py --explorer beta_gp_rloss --et svariance2 --seed $seed --width 3 > run.log 2>&1 &
#         python "$d".py --explorer beta_gp_rloss --et svariance2 --seed $seed --width 2 > run.log 2>&1 &
#         python "$d".py --explorer beta_gp_rloss --et svariance2 --seed $seed --width 5 > run.log 2>&1 &
#         python "$d".py --explorer beta_gp_rloss --et random --seed $seed --width 1 > run.log 2>&1 &
#         python "$d".py --explorer beta_gp_rloss --et random --seed $seed --width 3 > run.log 2>&1 &
#         python3.6 "$d".py --explorer bern_gp --et svariance2 --ft simple --seed $seed > run.log 2>&1 &
#         nohup python "$d".py --explorer beta_gp_rloss --et random --seed $seed --approx_type baseline > runs/run1_"$d"_"$seed".log 2>&1 &
#         nohup python "$d".py --explorer beta_gp_rloss --et svariance2 --seed $seed --approx_type baseline --no_scale_loss > runs/run1_"$d"_"$seed".log 2>&1 &
#         nohup python "$d".py --explorer beta_gp_rloss --et svariance2 --seed $seed --approx_type baseline > runs/run2_"$d"_"$seed".log 2>&1 &
        
#         nohup python "$d".py --explorer beta_gp_rloss --et svariance2 --seed $seed --approx_type simplev3 --no_scale_loss > run.log 2>&1 &
#         nohup python "$d".py --explorer beta_gp_rloss --et svariance2 --seed $seed --approx_type simplev3 > runs/run1_"$d"_"$seed".log 2>&1 &
#         python "$d".py --explorer simple --seed $seed --et svariance2 > run.log 2>&1 &
#         python "$d".py --explorer simple --seed $seed --et random > run.log 2>&1 &
        :
    done
done

# Estimation on the same random set of examples (ablation)
for d in celeba celeba_private cocos3 cocos3_10k; 
do
    for seed in 0 1 2;
    do
#         python "$d".py --explorer simple --seed $seed --ablation > run.log 2>&1 &
#         python "$d".py --explorer bern_gp_rloss --width 1 --seed $seed --ablation > run.log 2>&1 &
#         python "$d".py --explorer bern_gp2 --et svariance2 --seed $seed --width 1 --ablation > run.log 2>&1 &
#         python "$d".py --explorer bern_gp2 --et svariance2 --seed $seed --width 3 --ablation > run.log 2>&1 &

#         nohup python "$d".py --explorer beta_gp_rloss --seed $seed --ablation --approx_type rignore > run.log 2>&1 &
#         nohup python "$d".py --explorer beta_gp_rloss --seed $seed --ablation --alpha_beta --approx_type baseline --no_scale_loss > run.log 2>&1 &
#         nohup python "$d".py --explorer beta_gp_rloss --seed $seed --ablation --approx_type baseline --no_scale_loss --ablation_resume > run.log 2>&1 &
#         nohup python "$d".py --explorer beta_gp_rloss --seed $seed --ablation --nbr --approx_type simplev3 --ablation_resume > run.log 2>&1 &
#         nohup python "$d".py --explorer beta_gp_rloss --seed $seed --ablation --approx_type baseline > run.log 2>&1 &
#         nohup python "$d".py --explorer beta_gp_rloss --seed $seed --width 1 --ablation --approx_type baseline --dw_alpha 1 > run.log 2>&1 &
        :
    done
done

for d in celeba celeba_private cocos3 cocos3_10k; 
do
    for seed in 0 1 2;
    do
        python "$d".py --explorer simple --seed $seed --ablation > run.log 2>&1 &
        python "$d".py --explorer simple --seed $seed --ablation --sample_type correctednoep > run.log 2>&1 &
        python "$d".py --explorer simple --seed $seed --ablation --sample_type raw > run.log 2>&1 &
        :
    done
done

# region making ablation
for d in celeba_private;
do
#     for seed in 0 1 2;
#     do
#         for alpha in 1 0.25 -0.25 -1;
#         do 
#             python "$d".py --explorer beta_gp_rloss --et svariance2 --seed $seed --width 3 --freq_alpha $alpha > run.log 2>&1 &
#         done
#         python "$d".py --explorer beta_gp_rloss --et svariance2 --seed $seed --width 3 --nbr > run.log 2>&1 &
        :
#     done
done

# alpha-beta
for d in celeba_private cocos3 celeba cocos3_10k;
do
    for seed in 0 1 2;
    do
#         python "$d".py --explorer beta_gp_rloss --et svariance2 --seed $seed --ablation --width 1 --alpha_beta > run.log 2>&1 &
#         python "$d".py --explorer beta_gp_rloss --et svariance2 --seed $seed --ablation --width 3 --alpha_beta > run.log 2>&1 &
        :
    done
done    