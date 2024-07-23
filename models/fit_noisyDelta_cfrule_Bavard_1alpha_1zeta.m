function [out] = fit_noisyDelta_cfrule_Bavard_1alpha_1zeta(cfg)

% check parameters
if ~isfield(cfg,'cfrule')
    error('Missing counterfactual rule flag!');
elseif ~isscalar(cfg.cfrule) || ~islogical(cfg.cfrule)
    error('Invalid counterfactual rule flag!');
end
if ~isfield(cfg,'nstype')
    error('Missing noise type!');
elseif ~ismember(cfg.nstype,{'weber','white'})
    error('Invalid noise type!');
end
if ~isfield(cfg,'chrule')
    error('Missing choice rule!');
elseif ~ismember(cfg.chrule,{'thomp','softm'})
    error('Invalid choice rule!');
end
if ~isfield(cfg,'fitalgo')
    cfg.fitalgo = '';
elseif ~ismember(cfg.fitalgo,{'bads','vbmc'})
    error('Invalid fitting algorithm!');
end
if ~isfield(cfg,'noprior')
    error('Missing priors flag!');
end
if ~isfield(cfg,'nsmp')
    error('Missing number of samples used by particle filter!');
end
if ~isfield(cfg,'nres')
    cfg.nres = 1e2;
end
if ~isfield(cfg,'nrun')
    cfg.nrun = 1;
end
if ~isfield(cfg,'verbose')
    cfg.verbose = 0;
end

% get experiment data
if ~isfield(cfg,'cfg')
    trl    = cfg.trl;    % trial number in current block
    resp   = cfg.resp;   % responses
    rt     = cfg.rt;     % available rewards
    uopt   = cfg.uopt;   % which option is restricted (unavailable)
else
    trl    = cfg.cfg.trl;
    resp   = cfg.cfg.resp;
    rt     = cfg.cfg.rt;
end

% get total number of trials
ntrl = numel(trl);

nopt = sum(~isnan(nanmean(rt,1))); % number of bandits

% get fitting parameters
cfrule  = cfg.cfrule;  % counterfactual rule flag (true or false)
nstype  = cfg.nstype;  % noise type (weber or white)
chrule  = cfg.chrule;  % choice rule (thomp or softm)
fitalgo = cfg.fitalgo; % fitting algorithm (bads or vbmc)
noprior = cfg.noprior; % ignore priors?
nsmp    = cfg.nsmp;    % number of samples used by particle filter
nres    = cfg.nres;    % number of bootstrap/validation resamples
nrun    = cfg.nrun;    % number of random starting points (fitalgo = bads)
verbose = cfg.verbose; % fitting display level

% set internal parameters
epsi = 1e-6; % infinitesimal response probability

if cfrule
    % do not fit decay rate when counterfactual rule is used
    cfg.delta = nan;
end
% set fixed statistics
m0 = 0.5000; % prior mean

% define model parameters
pnam = cell(1,4); % name
pmin = nan(1,4);  % minimum value
pmax = nan(1,4);  % maximum value
pini = nan(1,4);  % initial value
pplb = nan(1,4);  % plausible lower bound
ppub = nan(1,4);  % plausible upper bound
pfun = cell(1,4); % prior function (empty = uniform)

% 1/ learning rate ~ uniform(0,1)
pnam{1} = 'alpha';
pmin(1) = 0.001;
pmax(1) = 0.999;
pini(1) = 0.5;
pplb(1) = 0.1;
ppub(1) = 0.9;
% 2/ decay rate ~ beta(1,9)
pnam{2} = 'delta';
pmin(2) = 0;
pmax(2) = 1;
pini(2) = 0.1;
pplb(2) = 0.01;
ppub(2) = 0.5;
if ~cfg.cfrule && isfield(cfg,'delta') && cfg.delta == 0
    pfun{2} = @(x)0; % added handling for delta = 0 while ~cfrule JL
else
    pfun{2} = @(x)betapdf(x,1,9);
end
% 3/ learning noise
pnam{3} = 'zeta';
switch nstype
    case 'weber' % Weber noise ~ exp(1)
        pmin(3) = 0;
        pmax(3) = 10;
        pini(3) = 1;
        pplb(3) = 0.1;
        ppub(3) = 5;
        pfun{3} = @(x)exppdf(x,1);
    case 'white' % white noise ~ exp(0.1)
        pmin(3) = 0;
        pmax(3) = 1;
        pini(3) = 0.1;
        pplb(3) = 0.01;
        ppub(3) = 0.5;
        pfun{3} = @(x)exppdf(x,0.1);
end
% 5/ policy temperature
pnam{4} = 'tau';
switch chrule
    case 'thomp' % Thompson sampling ~ exp(1)
        pmin(4) = 1e-12;
        pmax(4) = 10;
        pini(4) = 1;
        pplb(4) = 0.1;
        ppub(4) = 5;
        pfun{4} = @(x)exppdf(x,1);
    case 'softm' % softmax ~ exp(0.1)
        pmin(4) = 1e-12;
        pmax(4) = 1;
        pini(4) = 0.1;
        pplb(4) = 0.01;
        ppub(4) = 0.5;
        pfun{4} = @(x)exppdf(x,0.1);
end

% set number of parameters
npar = numel(pnam);

if noprior
    % ignore priors
    pfun = cell(1,npar);
end

% apply user-defined initialization values
if isfield(cfg,'pini')
    for i = 1:npar
        if isfield(cfg.pini,pnam{i}) && ~isnan(cfg.pini.(pnam{i}))
            pini(i) = cfg.pini.(pnam{i});
            % clamp initialization value within plausible bounds
            pini(i) = min(max(pini(i),pplb(i)+1e-6),ppub(i)-1e-6);
        end
    end
end

% define fixed parameters
pfix = cell(1,npar);
for i = 1:npar
    if isfield(cfg,pnam{i}) && ~isempty(cfg.(pnam{i}))
        pfix{i} = min(max(cfg.(pnam{i}),pmin(i)),pmax(i));
    end
end

% define free parameters
ifit = cell(1,npar);
pfit_ini = [];
pfit_min = [];
pfit_max = [];
pfit_plb = [];
pfit_pub = [];
n = 1;
for i = 1:npar
    if isempty(pfix{i}) % free parameter
        ifit{i} = n;
        pfit_ini = cat(2,pfit_ini,pini(i));
        pfit_min = cat(2,pfit_min,pmin(i));
        pfit_max = cat(2,pfit_max,pmax(i));
        pfit_plb = cat(2,pfit_plb,pplb(i));
        pfit_pub = cat(2,pfit_pub,ppub(i));
        n = n+1;
    end
end

% set number of fitted parameters
nfit = length(pfit_ini);

% determine whether agent is noisy
isnoisy = isempty(pfix{3}) || pfix{3} > 0;

if nfit > 0
    switch fitalgo
        
        case 'bads'
            % fit model using Bayesian Adaptive Direct Search
            if ~exist('bads','file')
                error('BADS missing from path!');
            end
            
            if ~noprior
                % do not use priors
                warning('Disabling the use of priors when using BADS.');
                % noprior = true;
                pfun = cell(1,npar);
            end

            % configure BADS
            options = bads('defaults');
            options.UncertaintyHandling = isnoisy; % noisy objective function
            options.NoiseFinalSamples = nres; % number of samples
            switch verbose % display level
                case 0, options.Display = 'none';
                case 1, options.Display = 'final';
                case 2, options.Display = 'iter';
            end
            
            % fit model using multiple random starting points
            fval   = nan(1,nrun);
            xhat   = cell(1,nrun);
            output = cell(1,nrun);
            for irun = 1:nrun
                done = false;
                while ~done
                    % set random starting point
                    n = 1;
                    for i = 1:npar
                        if isempty(pfix{i}) % free parameter
                            % sample starting point uniformly between plausible bounds
                            pfit_ini(n) = unifrnd(pplb(i),ppub(i));
                            n = n+1;
                        end
                    end
                    % fit model using BADS
                    [xhat{irun},fval(irun),exitflag,output{irun}] = ...
                        bads(@(x)getnl(x), ...
                        pfit_ini,pfit_min,pfit_max,pfit_plb,pfit_pub,[],options);
                    if exitflag > 0
                        done = true;
                    end
                end
            end
            % find best fit among random starting points
            [fval,irun] = min(fval);
            xhat   = xhat{irun};
            output = output{irun};
            
            % get best-fitting values
            phat = getpval(xhat);
            
            % create output structure with best-fitting values
            out = cell2struct(phat(:),pnam(:));
            
            % store fitting information
            out.fitalgo = fitalgo; % fitting algorithm
            out.nsmp    = nsmp;    % number of samples used by particle filter
            out.nres    = nres;    % number of validation resamples
            out.nrun    = nrun;    % number of random starting points
            out.ntrl    = ntrl;    % number of trials
            out.nfit    = nfit;    % number of fitted parameters
            
            % get maximum log-likelihood
            out.ll = -output.fval; % estimated log-likelihood
            out.ll_sd = output.fsd; % estimated s.d. of log-likelihood
            
            % get complexity-penalized fitting metrics
            out.aic = -2*out.ll+2*nfit+2*nfit*(nfit+1)/(ntrl-nfit+1); % AIC
            out.bic = -2*out.ll+nfit*log(ntrl); % BIC
            
            % get parameter values
            out.xnam = pnam(cellfun(@isempty,pfix));
            out.xhat = xhat;
            
            % store additional output from BADS
            out.output = output;
            
        case 'vbmc'
            % fit model using Variational Bayesian Monte Carlo
            if ~exist('vbmc','file')
                error('VBMC missing from path!');
            end
            
            % configure VBMC
            options = vbmc('defaults');
            options.MaxIter = 300; % maximum number of iterations
            options.MaxFunEvals = 500; % maximum number of function evaluations
            options.SpecifyTargetNoise = isnoisy; % noisy log-posterior function
            switch verbose % display level
                case 0, options.Display = 'none';
                case 1, options.Display = 'final';
                case 2, options.Display = 'iter';
            end
            
            % fit model using VBMC
            [vp,elbo,~,exitflag,output] = vbmc(@(x)getlp(x), ...
                pfit_ini,pfit_min,pfit_max,pfit_plb,pfit_pub,options);
            
            % generate 10^6 samples from the variational posterior
            xsmp = vbmc_rnd(vp,1e6);
            
            % get sample statistics
            xmap = vbmc_mode(vp); % posterior mode
            xavg = mean(xsmp,1); % posterior mean
            xstd = std(xsmp,[],1); % posterior s.d.
            xcov = cov(xsmp); % posterior covariance matrix
            xmed = median(xsmp,1); % posterior medians
            xqrt = quantile(xsmp,[0.25,0.75],1); % posterior 1st and 3rd quartiles
            
            % get full parameter set with best-fitting values
            phat_map = getpval(xmap); % posterior mode
            phat_avg = getpval(xavg); % posterior mean
            
            % use posterior mode as default
            phat = phat_map;
            
            % create output structure
            out = cell2struct(phat(:),pnam(:));
            
            % create substructure with posterior mode
            out.pmap = cell2struct(phat_map(:),pnam(:));
            
            % create substructure with posterior mean
            out.pavg = cell2struct(phat_avg(:),pnam(:));
            
            % store fitting information
            out.fitalgo = fitalgo; % fitting algorithm
            out.nsmp    = nsmp;    % number of samples used by particle filter
            out.nres    = nres;    % number of bootstrap resamples
            out.ntrl    = ntrl;    % number of trials
            out.nfit    = nfit;    % number of fitted parameters
            
            % store variational posterior solution
            out.vp = vp;
            
            % get ELBO (expected lower bound on log-marginal likelihood)
            out.elbo = elbo; % estimate
            out.elbo_sd = output.elbo_sd; % standard deviation
            
            % get maximum log-posterior and maximum log-likelihood
            out.lp = getlp(xmap); % log-posterior
            out.ll = getll(phat_map{:}); % log-likelihood
            
            % get parameter values
            out.xnam = pnam(cellfun(@isempty,pfix)); % fitted parameters
            out.xmap = xmap; % posterior mode
            out.xavg = xavg; % posterior mean
            out.xstd = xstd; % posterior s.d.
            out.xcov = xcov; % posterior covariance matrix
            out.xmed = xmed; % posterior median
            out.xqrt = xqrt; % posterior 1st and 3rd quartiles
            
            % store extra VBMC output
            out.output = output;
            
        otherwise
            error('Undefined fitting algorithm!');
            
    end
    
else
    % use fixed parameter values
    phat = getpval([]);
    
    % create output structure
    out = cell2struct(phat(:),pnam(:));
    
    % run particle filter
    [pt_hat,mt_hat,vt_hat,ut_hat,st_hat,wt_hat] = getp(phat{:});
    
    % average trajectories
    pt_avg = mean(pt_hat,3);
    mt_avg = sum(bsxfun(@times,mt_hat,reshape(wt_hat,[ntrl,1,nsmp])),3);
    ut_avg = sum(bsxfun(@times,ut_hat,reshape(wt_hat,[ntrl,1,nsmp])),3);
    
    % store averaged trajectories
    out.pt = pt_avg; % response probabilities
    out.mt = mt_avg; % filtered posterior means
    out.vt = vt_hat; % posterior variances
    
    % identify greedy responses
    [~,ru] = max(ut_avg,[],2); % based on exact updates
    [~,rf] = max(mt_avg,[],2); % based on noisy updates
    % compute fractions of non-greedy responses
    pngu = mean(resp ~= ru); % based on exact updates
    pngf = mean(resp ~= rf); % based on noisy updates
    pngd = mean(rf ~= ru); % due to noise
    pngx = mean(resp(resp ~= ru) == rf(resp ~= ru)); % explained by noise

    % store fractions of non-greedy responses
    out.pngu = pngu; % based on exact updates
    out.pngf = pngf; % based on noisy updates
    out.pngd = pngd; % due to noise
    out.pngx = pngx; % explained by noise
end

% store configuration structure
out.cfg = cfg;

    function [pval] = getpval(p)
        % get parameter values
        pval = cell(1,npar);
        for k = 1:npar
            if isempty(pfix{k}) % free parameter
                pval{k} = p(ifit{k});
            else % fixed parameter
                pval{k} = pfix{k};
            end
        end
    end

    function [nl] = getnl(p)
        % get parameter values
        pval = getpval(p);
        % get negative log-likelihood
        nl = -getll(pval{:});
    end

    function [lp,lp_sd] = getlp(p)
        % get parameter values
        pval = getpval(p);
        % get log-prior
        l0 = 0;
        for k = 1:npar
            if isempty(pfix{k}) % free parameter
                if isempty(pfun{k}) % use uniform prior
                    l0 = l0+log(unifpdf(pval{k},pmin(k),pmax(k)));
                else % use specified prior
                    l0 = l0+log(pfun{k}(pval{k}));
                end
            end
        end
        % get log-likelihood
        [ll,ll_sd] = getll(pval{:});
        % get log-posterior
        lp = ll+l0; % estimate
        lp_sd = ll_sd; % bootstrap s.d.
    end

    function [ll,ll_sd] = getll(varargin)
        % compute response probability
        p = getp(varargin{:});
        if nargout > 1
            % compute log-likelihood s.d.
            lres = nan(nres,1);
            for ires = 1:nres
                jres = randsample(nsmp,nsmp,true);
                pres = mean(p(:,:,jres),3);
                pres = adjust_probabilities_matrix(pres, epsi);
                nk = size(pres,1);
                pc = [];
                for k = nk:-1:1
                    pc(k,1) = pres(k,resp(k)); % chosen
                end
                lres(ires) = sum(log(pc));
            end
            ll_sd = max(std(lres),1e-6);
        end
        % compute log-likelihood
        p = mean(p,3);
        p = adjust_probabilities_matrix(p, epsi);
        nk = size(p,1);
        pc = [];
        for k = nk:-1:1
            pc(k,1) = p(k,resp(k)); % chosen
        end
        ll = sum(log(pc));
    end

    function [pt,mt,vt,ut,st,wt] = getp(alpha,delta,zeta,tau)
        % initialize output variables
        pt = nan(ntrl,nopt,nsmp); % response probabilities, (more general)
        mt = nan(ntrl,nopt,nsmp); % posterior means
        vt = nan(ntrl,nopt);      % posterior variances
        ut = nan(ntrl,nopt,nsmp); % filtering means
        st = nan(ntrl,nopt,nsmp); % filtering noise
        wt = nan(ntrl,nsmp);      % filtering weights
        et = nan(ntrl,nopt,nsmp); % prediction errors
        % run particle filter
        for itrl = 1:ntrl

            if trl(itrl) == 1
                % initialize posterior means and variance
                mt(itrl,:,:) = m0;
                % respond randomly
                pt(itrl,:,:) = 1/nopt;
                if uopt(itrl)
                    pt(itrl,uopt(itrl),:) = NaN; % option unavailable
                    pt(itrl,setxor(1:nopt,uopt(itrl)),:) = 1/(nopt-1);
                end
                % initialize filtering noise and weights
                st(itrl,:,:) = 0;
                wt(itrl,:) = 1/nsmp;
                continue
            end
            % condition posterior means using weighted bootstrapping
            mt(itrl,:,:) = mt(itrl-1,:,randsample(nsmp,nsmp,true,wt(itrl-1,:)));
            % update posterior means and variances
            c = resp(itrl-1);
            uall = setxor(1:nopt,c);
            et(itrl,c,:) = rt(itrl-1,c)-mt(itrl,c,:);
            if strcmp(nstype,'weber')
                st(itrl,c,:) = zeta*abs(et(itrl,c,:));
            else
                st(itrl,c,:) = zeta;
            end
            mt(itrl,c,:) = mt(itrl,c,:)+alpha*et(itrl,c,:);
            for whichu = 1:length(uall)
                u = uall(whichu);
                if cfrule % counterfactual rule
                    et(itrl,u,:) = rt(itrl-1,u)-mt(itrl,u,:);
                    if strcmp(nstype,'weber')
                        st(itrl,u,:) = zeta*abs(et(itrl,u,:));
                    else
                        st(itrl,u,:) = zeta;
                    end
                    mt(itrl,u,:) = mt(itrl,u,:)+alpha*et(itrl,u,:);
                else % decay
                    st(itrl,u,:) = 0;
                    mt(itrl,u,:) = mt(itrl,u,:)+delta*(m0-mt(itrl,u,:));
                end
            end
            % account for filtering noise
            ut(itrl,:,:) = mt(itrl,:,:); % store filtering means
            if isnoisy
                mt(itrl,c,:) = noisrnd(mt(itrl,c,:),st(itrl,c,:));
                if cfrule % counterfactual rule
                    for whichu = 1:length(uall)
                        u = uall(whichu);
                        mt(itrl,u,:) = noisrnd(mt(itrl,u,:),st(itrl,u,:));
                    end
                end
            end
            % apply policy to get response probabilities
            q = mt(itrl,:,:)/tau;
            if uopt(itrl)
                q(:,uopt(itrl),:) = NaN; % best option unavailable
            end
            % softmax:
            pt(itrl,:,:) = exp(q-prctile(q,100,2))...
                           ./nansum(exp(q-prctile(q,100,2)),2);
           
            % compute filtering weights
            wt(itrl,:) = pt(itrl,resp(itrl),:);
            if nnz(wt(itrl,:)) == 0
                wt(itrl,:) = 1/nsmp;
            else
                wt(itrl,:) = wt(itrl,:)/sum(wt(itrl,:));
            end
        end
    end

end