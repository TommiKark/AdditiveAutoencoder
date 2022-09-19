function out = Lbfgsc1(FUN,x0,varargin)
%LBFGS   Limited-memory BFGS minimization (vector-based).
%
%  OUT = LBFGS(FUN,X0) minimizes FUN starting at the point X0 using L-BFGS.
%  FUN is a handle for a function that takes a single vector input and
%  returns two arguments --- the scalar function value and the
%  vector-valued gradient. See POBLANO_OUT for details of the output
%  parameters.
%
%  OUT = LBFGS(FUN,X0,'param',value,...) specifies a parameters and its
%  value. See POBLANO_PARAMS for further details on standard parameters.
%  Additionally, LBFGS requires
%
%   'M' - Limited memory parameter {5}.
%
%  PARAMS = LBFGS('defaults') returns a structure containing the 
%  default parameters for the particular Poblano method. 
%
%
%  Examples 
%  
%  Suppose the function and gradient of the objective function are
%  specified in an mfile named example1.m:
%
%    function [f,g]=example1(x,a)
%    if nargin < 2, a = 1; end
%    f = sin(a*x);
%    g = a*cos(a*x);
%
%  We can call the optimization method (using its default
%  parameters) using the command:
%
%    out = lbfgs(@(x) example1(x,3), pi/4);
%
%  To change a parameter, we can specify a param/value input pair
%  as follows:
%
%    out = lbfgs(@(x) example1(x,3), pi/4, 'Display', 'final');
%
%  Alternatively, we can use a structure to define the parameters:
%  
%    params.MaxIters = 2;
%    out = lbfgs(@(x) example1(x,3), pi/4, params);
%
%  See also POBLANO_OUT, POBLANO_PARAMS, FUNCTION_HANDLE.
%
%Poblano Toolbox for MATLAB
%
%Copyright 2009 National Technology & Engineering Solutions of Sandia,
%LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
%U.S. Government retains certain rights in this software.
%
% The original Lbfgs version from Poblano was modified by T. Kärkkäinen on 
% April, 2020, as follows (to be structurally modifiable for sampling based 
% algorithmic variants):
% - First all routines from Lbfgs.m + poblano_linesearch.m + cvsrch.m were
%   moved into one m-file and the poblano_linesearch wrapper was removed
% - Then all code was further restructured as one main function and one
%   function call to the line search.
% - Then call to line search was modified for the minibatch version in
%   the file DsLbfgsc1.m.

%% Parse parameters

% Create parser
params = inputParser;

% Set Poblano parameters
params = poblano_params(params);

% Set parameters for this method
params.addParamValue('M',5,@(x) x > 0);

% Parse input
params.parse(varargin{:});

%% Check input arguments
if (nargin == 1) && isequal(FUN,'defaults') && (nargout == 1)
    out = params.Results;
    return;
elseif (nargin < 2)
    error('Error: invalid input arguments');
end

%% Initialize

xk = x0;
[fk,gk] = feval(FUN,xk);

out = poblano_out(xk,fk,gk,1,params);

%% Main loop
while out.ExitFlag == -1

    if out.Iters == 0
        % Initialize quantities before first iteration
        pk = -gk;
        ak = 1.0;
        S = [];
        Y = [];
        rho = [];
    else
        % Precompute quantites used in this iteration
        sk = xk - xkold;
        yk = gk - gkold;
        skyk = yk'*sk;
        ykyk = yk'*yk;
        rhok = 1 / skyk;
        gamma = skyk/ykyk;

        % Use information from last M iterations only
        if out.Iters <= params.Results.M
            S = [sk S];
            Y = [yk Y];
            rho = [rhok rho];
        else
            S = [sk S(:,1:end-1)];
            Y = [yk Y(:,1:end-1)];
            rho = [rhok rho(1:end-1)];
        end
        
        % Adjust M to available number of iterations
        m = size(S,2);

        % L-BFGS two-loop recursion
        q = gk;        
        for i = 1:m
            alpha(i) = rho(i)*S(:,i)'*q;
            q = q - alpha(i)*Y(:,i);
        end
        r = gamma*q;
        for i = m:-1:1
            beta = rho(i)*Y(:,i)'*r;
            r = r + (alpha(i) - beta)*S(:,i);
        end
        
        % r contains H_k * g_k (Hessian approximation at iteration k times
        % the gradient at iteration k
        pk = -r;
    end
    xkold = xk;
    gkold = gk;

    % Compute step length
    
    % Check whether user specified an initial step
    if params.Results.LineSearch_initialstep > 0
        ak = params.Results.LineSearch_initialstep;
    end
    [xk,fk,gk,ak,lsinfo,nfev] = cvsrchinc(FUN,xk,fk,gk,ak,pk,params.Results);

    if (lsinfo ~= 1) && strcmp(params.Results.Display, 'iter')
        fprintf(1,[mfilename,': line search warning = %d\n'],lsinfo);
    end
    
    % Update counts, check exit conditions, etc.
    out = poblano_out(xk,fk,gk,nfev,params,out);    
end


function [x,f,g,stp,info,nfev] = cvsrchinc(fcn,x,f,g,stp,s,params)
%CVSRCH   More-Thuente line search from MINPACK.

% Additions from D. Dunlavy
n = length(x);

% Initialize params
xtol = params.LineSearch_xtol;
ftol = params.LineSearch_ftol;
gtol = params.LineSearch_gtol;
stpmin = params.LineSearch_stpmin;
stpmax = params.LineSearch_stpmax;
maxfev = params.LineSearch_maxfev;

% move up in case of early termination (before this was initialized
nfev = 0;

% Start of D. O'Leary translation
p5 = .5;
p66 = .66;
xtrapf = 4;
info = 0;
infoc = 1;
%
%     Check the input parameters for errors.
%
if (n <= 0 | stp <= 0.0 | ftol < 0.0 |  ...
        gtol < 0.0 | xtol < 0.0 | stpmin < 0.0  ...
        | stpmax < stpmin | maxfev <= 0)
    return
end
%
%     Compute the initial gradient in the search direction
%     and check that s is a descent direction.
%
dginit = g'*s;
if (dginit >= 0.0)
    return
end
%
%     Initialize local variables.
%
brackt = 0;
stage1 = 1;
% moved up to initialize before any potential return
% nfev = 0;
finit = f;
dgtest = ftol*dginit;
width = stpmax - stpmin;
width1 = 2*width;
wa = x;
%
%     The variables stx, fx, dgx contain the values of the step,
%     function, and directional derivative at the best step.
%     The variables sty, fy, dgy contain the value of the step,
%     function, and derivative at the other endpoint of
%     the interval of uncertainty.
%     The variables stp, f, dg contain the values of the step,
%     function, and derivative at the current step.
%
stx = 0.0;
fx = finit;
dgx = dginit;
sty = 0.0;
fy = finit;
dgy = dginit;
%
%     Start of iteration.
%
while (1)
    %
    %        Set the minimum and maximum steps to correspond
    %        to the present interval of uncertainty.
    %
    if (brackt)
        stmin = min(stx,sty);
        stmax = max(stx,sty);
    else
        stmin = stx;
        stmax = stp + xtrapf*(stp - stx);
    end
    %
    %        Force the step to be within the bounds stpmax and stpmin.
    %
    stp = max(stp,stpmin);
    stp = min(stp,stpmax);
    %
    %        If an unusual termination is to occur then let
    %        stp be the lowest point obtained so far.
    %
    if ((brackt & (stp <= stmin | stp >= stmax)) ...
            | nfev >= maxfev-1 | infoc == 0 ...
            | (brackt & stmax-stmin <= xtol*stmax))
        stp = stx;
    end
    %
    %        Evaluate the function and gradient at stp
    %        and compute the directional derivative.
    %
    x = wa + stp * s;
    [f,g] = feval(fcn,x);
    nfev = nfev + 1;
    dg = g' * s;
    ftest1 = finit + stp*dgtest;
    %
    %        Test for convergence.
    %
    if ((brackt & (stp <= stmin | stp >= stmax)) | infoc == 0)
        info = 6;
    end
    if (stp == stpmax & f <= ftest1 & dg <= dgtest)
        info = 5;
    end
    if (stp == stpmin & (f > ftest1 | dg >= dgtest))
        info = 4;
    end
    if (nfev >= maxfev)
        info = 3;
    end
    if (brackt & stmax-stmin <= xtol*stmax)
        info = 2;
    end
    if (f <= ftest1 & abs(dg) <= gtol*(-dginit))
        info = 1;
    end
    %
    %        Check for termination.
    %
    if (info ~= 0)
        return
    end
    %
    %        In the first stage we seek a step for which the modified
    %        function has a nonpositive value and nonnegative derivative.
    %
    if (stage1 & f <= ftest1 & dg >= min(ftol,gtol)*dginit)
        stage1 = 0;
    end
    %
    %        A modified function is used to predict the step only if
    %        we have not obtained a step for which the modified
    %        function has a nonpositive function value and nonnegative
    %        derivative, and if a lower function value has been
    %        obtained but the decrease is not sufficient.
    %
    if (stage1 & f <= fx & f > ftest1)
        %
        %           Define the modified function and derivative values.
        %
        fm = f - stp*dgtest;
        fxm = fx - stx*dgtest;
        fym = fy - sty*dgtest;
        dgm = dg - dgtest;
        dgxm = dgx - dgtest;
        dgym = dgy - dgtest;
        %
        %           Call cstep to update the interval of uncertainty
        %           and to compute the new step.
        %
        [stx,fxm,dgxm,sty,fym,dgym,stp,fm,dgm,brackt,infoc] ...
            = cstep(stx,fxm,dgxm,sty,fym,dgym,stp,fm,dgm, ...
            brackt,stmin,stmax);
        %
        %           Reset the function and gradient values for f.
        %
        fx = fxm + stx*dgtest;
        fy = fym + sty*dgtest;
        dgx = dgxm + dgtest;
        dgy = dgym + dgtest;
    else
        %
        %           Call cstep to update the interval of uncertainty
        %           and to compute the new step.
        %
        [stx,fx,dgx,sty,fy,dgy,stp,f,dg,brackt,infoc] ...
            = cstep(stx,fx,dgx,sty,fy,dgy,stp,f,dg, ...
            brackt,stmin,stmax);
    end
    %
    %        Force a sufficient decrease in the size of the
    %        interval of uncertainty.
    %
    if (brackt)
        if (abs(sty-stx) >= p66*width1)
            stp = stx + p5*(sty - stx);
        end
        width1 = width;
        width = abs(sty-stx);
    end
    %
    %        End of iteration.
    %
end
%
%     Last card of subroutine cvsrch.
%
