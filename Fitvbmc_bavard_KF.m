clear; clc; close all
maindir = '/mnt/homes/home030/yinancao/RLsampling2023/paperRepo/bavard_modelfit/';
cd(maindir)
% add directories
addpath(genpath('./datasets'));
addpath(genpath('./toolbox'));
addpath(genpath('./models'));

load('Bavard_data_learning.mat');

norm_flag = 0; % default 0, just take the raw feedback
norm_str  = '';
if norm_flag
   norm_str = 'range_norm';
end
nfit_bads = 3;
ns        = length(Data);
k = 1; Data_stack = cell(0);
for s = 1:ns
    data = Data{s};
    for c = 1:size(data,3)
        Data_stack{k,1} = data(:,:,c);
        k = k + 1;
    end
end
size(Data_stack)

for s = 1:length(Data_stack)
    
    data        = Data_stack{s};
    data_raw    = [data,nan(size(data,1),6)];
    miss        = isnan(data_raw(:,6));
    data_unit   = data_raw(~miss,:); % remove miss trial
    ntrl        = size(data_unit,1);
    reward      = data_unit(:,3:5);
    nopt        = sum(~isnan(nanmean(reward,1)));
    reward      = reward(:,1:nopt);
    v_range     = [1,99];
    reward      = (reward-v_range(1))/(v_range(2)-v_range(1)); % rescale
    
    if norm_flag
    reward      = (reward-min(reward,[],2))./(max(reward,[],2)-min(reward,[],2));
    end
    
    cfg         = [];
    cfg.resp    = data_unit(:,6); % reward
    cfg.rt      = reward;         % option 1,2,3
    cfg.trl     = (1:ntrl)';      % trial number
    cfg.cxt     = data_unit(:,2); % context id
    cfg.respT   = data_unit(:,7); % reaction time
    uopt        = zeros(size(data_unit,1),1);
    uopt(data_unit(:,2)<0) = 1; % the best can be unavailable
    cfg.uopt    = uopt;
   
    cfg.cfrule  = true;  % counterfactual rule flag (true or false)
    cfg.nstype  = 'weber';  % noise type (weber or white)
    cfg.chrule  = 'softm';  % choice rule (thomp or softm)
    cfg.fitalgo = 'bads'; % fitting algorithm (bads or vbmc)
    cfg.noprior = 1; % ignore priors?
    cfg.nsmp    = 1e3;  % number of samples used by particle filter
    cfg.nres    = 1e2;  % number of bootstrap/validation resamples
    cfg.nrun    = nfit_bads; % number of random starting points (fitalgo = bads)
    cfg.verbose = 2; % fitting display level
    cfg.norm    = 0;
    
    fitfun       = @(x)fit_noisyKF_cfrule_Bavard_1zeta(x);
    saveout{s}{1} = getvbmc(cfg,fitfun);
    fitfun       = @(x)fit_noisyKF_cfrule_Bavard_2zeta(x);
    saveout{s}{2} = getvbmc(cfg,fitfun);
    
end

time_str = strrep(mat2str(fix(clock)),' ','_');
save(['Bavard_KF',norm_str,'_',time_str,'.mat'],'saveout','nfit_bads')
% exit;
%%

function saveout = getvbmc(cfg,fitfun)

    bads_fit     = fitfun(cfg);
    cfg2         = cfg;
    cfg2.fitalgo = 'vbmc'; % fitting algorithm (bads or vbmc)
    cfg2.noprior = 0;
    cfg2.nrun    = 1;
    xnam         = bads_fit.xnam;
    for pk = 1:length(xnam)
        field_name = xnam{pk};
        cfg2.pini.(field_name) = bads_fit.(field_name);
    end
    vbmc_fit     = fitfun(cfg2);
   for pk = 1:length(xnam)
        field_name = xnam{pk};
        % use the posterior mean
        cfg2.(field_name) = vbmc_fit.pavg.(field_name);
   end
    vbmc_pred    = fitfun(cfg2);
    % this has everything we need, including raw data input
    saveout = vbmc_pred;
    saveout.vbmc_info = rmfield(vbmc_fit,'cfg');
    saveout.bads_info = rmfield(bads_fit,'cfg');
    
end