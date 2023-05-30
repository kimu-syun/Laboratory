function ADEM_talk

% ADEM_learningを基にした実行ファイル
 
 
% generative model
% 生成モデル
%==========================================================================
rng('shuffle')
DEMO     = 2;                           % switch for demo

G(1).E.s = 1/2;                         % smoothness
G(1).E.n = 4;                           % smoothness
G(1).E.d = 2;                           % smoothness
 
% parameters
% パラメータ
%--------------------------------------------------------------------------
P.a     = 0;
P.b     = [0 0];
P.c     = [0 0 0 0];
P.d     = 0;
P0      = P;
pC      = speye(length(spm_vec(P)));
pC(end) = 0;
 
% level 1
%--------------------------------------------------------------------------
G(1).x  = [0; 0];
G(1).f  = 'spm_fx_talk';
G(1).g  = @(x,v,a,P) x;
G(1).pE = P;
G(1).pC = pC;
G(1).V  = exp(8);                       % error precision
G(1).W  = exp(8);                       % error precision
 
% level 2
%--------------------------------------------------------------------------
G(2).a  = 0;                            % action
G(2).v  = 0;                            % inputs
G(2).V  = exp(16);
G       = spm_ADEM_M_set(G);
 
 
% desired equilibrium density and state space
% 望ましい平衡密度と状態空間
%==========================================================================
G(1).fq = 'spm_mountaincar_Q';
 
% create X - coordinates of evaluation grid
% 評価グリッドの座標を作成
%--------------------------------------------------------------------------
nx      = 32;
x{1}    = linspace(-1,1,nx);
x{2}    = linspace(-1,1,nx);
[X,x]   = spm_ndgrid(x);
G(1).X  = X;
 
 
% optimise parameters so that p(y|G) maximises the cost function
% p(y|G)がコスト関数を最大化するようにパラメータを最適化
%==========================================================================
 
% optimise parameters: (NB an alternative is P = spm_fp_fmin(G));
% パラメータを最適化する
%--------------------------------------------------------------------------
if DEMO
    
    % Optimise parameters of fictive forces using KL control
    % KL制御による架空の力のパラメータ制御
    %----------------------------------------------------------------------
    P.a = 0.1;
    P.b = [0.2 0.2];
    P.c = [0.1 0.32 0.3 0.4];
    P.d = 0;
    
    if DEMO > 1
        P   = spm_fmin('spm_talk_fun',P,pC,G);
        P.d = 0;
    end
    
    G(1).pE = P;
    disp(P)
    save spm_talk_Q_model.mat G
end
 
 
% or load previously optimised environment
% また、事前に最適化された環境をロードする
%--------------------------------------------------------------------------
%{
load mountaincar_model
P     = G(1).pE;
%}
% plot flow fields and nullclines
% フローフィールドとヌルクラインをプロット
%==========================================================================
spm_figure('GetWin','Figure 1');
 
nx    = 64;
x{1}  = linspace(-2,2,nx);
x{2}  = linspace(-2,2,nx);
M     = G;
 
% uncontrolled flow (P0)
% 制御なしの流れ
%--------------------------------------------------------------------------
M(1).pE = P0;
subplot(3,2,1)
spm_fp_display_density(M,x);
xlabel('emotions','Fontsize',12)
ylabel('rilationship','Fontsize',12)
title('flow and equilibrium density','Fontsize',16)
 
subplot(3,2,2)
spm_fp_display_nullclines(M,x);
xlabel('emotions','Fontsize',12)
ylabel('rilationship','Fontsize',12)
title('nullclines','Fontsize',16)
 
% controlled flow (P0)
% 制御ありの流れ
%--------------------------------------------------------------------------
M(1).pE = P;
subplot(3,2,3)
spm_fp_display_density(M,x);
xlabel('emotions','Fontsize',12)
ylabel('rilationship','Fontsize',12)
title('controlled','Fontsize',16)
 
subplot(3,2,4)
spm_fp_display_nullclines(M,x);
xlabel('emotions','Fontsize',12)
ylabel('rilationship','Fontsize',12)
title('controlled','Fontsize',16)
drawnow
 

% recognition model: learn the controlled environmental dynamics
% 識別モデル：制御された環境ダイナミクスの学習
%==========================================================================
%{
M       = G;
M(1).g  = @(x,v,P)x;
 
% make a niave model (M)
% ニーブモデルを作る(niave 素朴な)
%--------------------------------------------------------------------------
M(1).pE = P0;
M(1).pC = exp(8);
M(1).V  = exp(8);
M(1).W  = exp(8);

% teach naive model by exposing it to a controlled environment (G)
% 制御された環境(G)をさらすことで素朴なモデルを教える
%--------------------------------------------------------------------------
clear DEM
 
% perturbations
% 攪乱
%--------------------------------------------------------------------------
n     = 16;
i     = (1:n)*32;
C     = sparse(1,i,randn(1,n));
C     = spm_conv(C,4);
 
DEM.M = M;
DEM.G = G;
DEM.C = C;
DEM.U = C;
 
% optimise recognition model
% 識別モデルの最適化
%--------------------------------------------------------------------------
if DEMO
    DEM.M(1).E.nE = 16;
    DEM           = spm_ADEM(DEM);
    save mountaincar_model G DEM
end

load mountaincar_model

 
% replace priors with learned conditional expectation
% priorを学習した条件付き期待値に置き換える
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 1');
 
M(1).pE = DEM.qP.P{1};
M(1).pC = [];

subplot(3,2,5)
spm_fp_display_density(M,x);
xlabel('position','Fontsize',12)
ylabel('velocity','Fontsize',12)
title('learnt','Fontsize',16)
 
subplot(3,2,6)
spm_fp_display_nullclines(M,x);
xlabel('position','Fontsize',12)
ylabel('velocity','Fontsize',12)
title('learnt','Fontsize',16)

 
% evaluate performance under active inference
% 能動的推論のもとで性能を評価する
%==========================================================================
 
% create uncontrolled environment (with action)
% 非制御の環境を作る(行動と)
%--------------------------------------------------------------------------
G(1).pE   = P0;
G(1).pE.d = 1;
G(1).U    = exp(8);
G(1).V    = exp(16);
G(1).W    = exp(16);

% create DEM structure (and perturb the real car with fluctuations)
% DEM構造を作る(実車に揺らぎを与える)
%--------------------------------------------------------------------------
N       = 128;
U       = sparse(1,N);
C       = spm_conv(randn(1,N),8)/4;      % pertubations
DEM.G   = G;
DEM.M   = M;
DEM.C   = U;
DEM.U   = U;
DEM     = spm_ADEM(DEM);
 
% overlay true values
% オーバーレイ真値
%--------------------------------------------------------------------------
spm_figure('GetWin','DEM');
spm_DEM_qU(DEM.qU,DEM.pU)
 
subplot(2,2,3)
spm_fp_display_nullclines(M,x);hold on
plot(DEM.pU.v{1}(1,:),DEM.pU.v{1}(2,:),'b'), hold off
xlabel('position','Fontsize',12)
ylabel('velocity','Fontsize',12)
title('learnt','Fontsize',16)

% movie
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 2');
clf, subplot(3,1,2)
drawnow
spm_mountaincar_movie(DEM)
title('click car for movie','FontSize',16)
%}


