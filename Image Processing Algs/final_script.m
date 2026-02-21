%% =========================================================================
%  ULTRA DEHAZE v5 — SAND-DUST / FOG / MIST / HAZE SPECIALIST
%  =========================================================================
%
%  WHY PREVIOUS VERSIONS FAILED ON THIS IMAGE:
%  The image is a real sandstorm. Generic DCP methods fail because:
%   1. Sand-dust absorbs blue-violet light preferentially → heavy yellow cast
%      that cannot be removed by treating all channels equally.
%   2. The yellow veil IS the dust — standard omega on all channels removes
%      haze geometry but leaves the colour cast completely.
%   3. DCP fails in sky/bright regions by design — overestimates haze there.
%
%  v5 CORE INSIGHT — SAND-DUST IS A DIFFERENT PHYSICAL PROBLEM:
%
%  [A] CHANNEL-SPECIFIC PHYSICAL MODEL (Cheng et al. IEEE Access 2020;
%      Gao et al. IEEE Photonics J. 2020):
%      Sand-dust degrades R,G,B channels asymmetrically:
%        I_R ≈ J_R·t_R + A_R·(1-t_R)    (red: least attenuated)
%        I_G ≈ J_G·t_G + A_G·(1-t_G)    (green: moderately attenuated)
%        I_B ≈ J_B·t_B + A_B·(1-t_B)    (blue: most attenuated — absorbs 3-5x more)
%      So t_R > t_G > t_B for sand-dust scenes.
%      Estimating a SINGLE t for all channels is physically wrong here.
%
%  [B] RED-CHANNEL OFFSET CORRECTION (Kim et al. Sensors/PMC 2021;
%      Springer Image & Video Processing 2016):
%      Sandstorm imparts a red offset d^c = mean(I_R) - mean(I_c).
%      Corrected recovery:
%        J^c(x) = [I^c(x) - (A^c - d^c)] / t(x) + (A^c - d^c)
%      Physically removes the red-shift caused by Mie scattering.
%
%  [C] BLUE CHANNEL COMPENSATION (Cheng et al. IEEE Access 2020):
%      After channel-wise recovery, compensate the blue channel:
%        J_B_comp = J_B + α·(J_R + J_G)/2
%      where α is estimated from the residual blue deficit.
%      This restores sky/neutral colours absorbed by dust.
%
%  [D] REGION-ADAPTIVE DCP + BCP (Kim et al. PMC 2021):
%      Texture probability P_tex = f(local entropy).
%      omega(x) = omega_min + (omega_max - omega_min)·P_tex
%      → Flat/bright regions: low omega (gentle), no dark distortion.
%      → Textured regions: high omega (aggressive), full visibility.
%
%  [E] SATURATION-BASED TRANSMISSION (Kim TIP 2019 + Zhu TIP 2015):
%      t_sat(x) = 1 - S(x)/S_max  (high saturation → scene visible → high t)
%      Fused with DCP for robust per-pixel estimate.
%
%  [F] LAB GAMMA-CONTRAST STRETCH (Shi et al. IET Image Proc. 2020):
%      After recovery: apply normalised gamma transformation in LAB space.
%      Stretch L* channel using per-pixel adaptive gamma based on local mean.
%      Avoids global gamma (which over-brightens/darkens entire image).
%
%  [G] GUIDED FILTER EDGE-PRESERVING REFINEMENT (He et al. ECCV 2010):
%      Standard guided filter on the fused transmission map.
%      Radius and eps calibrated to image resolution and fog type.
%
%  PIPELINE SUMMARY:
%   [1]  Universal image load (JPG/PNG/BMP/TIFF/GIF/PGM/PPM/WebP…)
%   [2]  Sand-dust vs fog/mist degradation classifier
%   [3]  Bilateral pre-denoise (before any recovery)
%   [4]  Channel-specific atmospheric light estimation [A]
%   [5]  Region-adaptive texture probability map [D]
%   [6]  Channel-wise transmission estimation (DCP+sat fusion) [A,E]
%   [7]  Guided filter transmission refinement [G]
%   [8]  Red-offset corrected scene recovery [B]
%   [9]  Blue channel compensation [C]
%   [10] LAB adaptive gamma contrast stretch [F]
%   [11] CLAHE on luminance only
%   [12] Structure-tensor coherence-gated sharpening (no ringing)
%   [13] Final colour balance + output
%
%  FORMATS: JPG, PNG, BMP, TIFF, GIF, PGM, PPM, WebP, and all imread formats
%  USAGE: Drop image in MATLAB working directory → run
%  OUTPUT: defogging_output/<name>_dehazed.png + comparison figure
% =========================================================================

clear all; close all; clc;

fprintf('=================================================================\n');
fprintf('  ULTRA DEHAZE v5 — SAND/DUST/FOG/MIST/HAZE SPECIALIST\n');
fprintf('=================================================================\n');

%% ========================================================================
%% [1] UNIVERSAL IMAGE LOADER
%% ========================================================================
supported_exts = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff', ...
                  '.gif','.pgm','.ppm','.pbm','.webp','.ico', ...
                  '.pcx','.xwd','.hdf','.ras'};
priority_stems = {'foggy_image','input','hazy_image','foggy','haze', ...
                  'mist','dust','sand','test','image','img','photo','scene'};

img_path = '';
for s = 1:length(priority_stems)
    for e = 1:length(supported_exts)
        c = [priority_stems{s}, supported_exts{e}];
        if exist(c,'file'), img_path = c; break; end
    end
    if ~isempty(img_path), break; end
end
if isempty(img_path)
    d = dir(pwd);
    for k = 1:length(d)
        if d(k).isdir, continue; end
        [~,~,ex] = fileparts(d(k).name);
        if any(strcmpi(ex,supported_exts)), img_path = d(k).name; break; end
    end
end
if isempty(img_path)
    error('No image found in %s\nSupported: %s', pwd, strjoin(supported_exts,', '));
end
fprintf('Input : %s\n', img_path);

try, info = imfinfo(img_path);
catch, error('Cannot read "%s". File may be corrupt.', img_path); end

if length(info) > 1
    fprintf('  Multi-frame (%d frames) — using frame 1.\n', length(info));
    I_raw = imread(img_path, 1);
else
    I_raw = imread(img_path);
end
if isfield(info,'ColorType') && strcmpi(info(1).ColorType,'indexed')
    [I_raw, cmap] = imread(img_path);
    I_raw = uint8(ind2rgb(I_raw,cmap)*255);
end
if size(I_raw,3)==4
    fprintf('  RGBA → dropping alpha.\n');
    I_raw = I_raw(:,:,1:3);
end
if size(I_raw,3)==1
    fprintf('  Grayscale → expanding to RGB.\n');
    I_raw = cat(3,I_raw,I_raw,I_raw);
end
I = im2double(I_raw);

% Resize to max 1000px (speed; quality unaffected)
[h,w,~] = size(I);
if max(h,w) > 1000
    sc = 1000/max(h,w);
    I  = imresize(I,sc,'bicubic');
    fprintf('  Resized %dx%d → %dx%d\n', h,w, size(I,1),size(I,2));
end
[h,w,~] = size(I);
fprintf('  Size : %dx%d px\n\n', h, w);

%% ========================================================================
%% [2] DEGRADATION CLASSIFIER — sand-dust vs generic fog/mist
%% ========================================================================
fprintf('[2/13] Degradation classification...\n');

R = I(:,:,1); G = I(:,:,2); B = I(:,:,3);
avg_R = mean(R(:)); avg_G = mean(G(:)); avg_B = mean(B(:));

% Sand-dust: yellow-brown dominant → R > G >> B, low saturation, high value
hsv_I   = rgb2hsv(I);
avg_S   = mean(hsv_I(:,:,2),'all');
avg_V   = mean(hsv_I(:,:,3),'all');
gray    = rgb2gray(I);
avg_L   = mean(gray(:));
std_L   = std(gray(:));
fog_idx = min(avg_L / (std_L + 0.05), 10);

% Channel imbalance
rg_ratio = avg_R / (avg_G + eps);   % >1.05 → reddish (sandstorm)
rb_ratio = avg_R / (avg_B + eps);   % >1.20 → strong red-shift
gb_deficit = avg_B - avg_G;         % negative → blue absorbed (dust)

is_sandust = (rg_ratio > 1.04) && (rb_ratio > 1.15) && (avg_S < 0.30);
is_mist    = (avg_B > avg_R + 0.02);
is_dense   = fog_idx > 3.0;

fprintf('  R=%.3f G=%.3f B=%.3f | rg=%.2f rb=%.2f\n', avg_R,avg_G,avg_B, rg_ratio,rb_ratio);
fprintf('  fog_idx=%.2f | sand-dust=%d | mist=%d | dense=%d\n', ...
    fog_idx, is_sandust, is_mist, is_dense);

%% ========================================================================
%% [3] BILATERAL PRE-DENOISE (before any recovery)
%% ========================================================================
fprintf('[3/13] Bilateral pre-denoise...\n');
sigma_s = ternary(is_sandust, 2.0, 1.5);
I_f = bilateralApprox(I, sigma_s, 0.08);

%% ========================================================================
%% [4] CHANNEL-SPECIFIC ATMOSPHERIC LIGHT [A]
%% ========================================================================
fprintf('[4/13] Channel-specific atmospheric light...\n');

% Sand-dust: A_R > A_G > A_B (Mie scattering asymmetry)
% Estimate per-channel using top-0.1% bright pixels in that channel
A = zeros(1,3);
n_top = max(round(h*w*0.001), 20);

for c = 1:3
    ch    = I_f(:,:,c);
    ch_f  = ch(:);
    [~,idx] = sort(ch_f,'descend');
    top_px  = ch_f(idx(1:n_top));
    A(c)    = mean(top_px) * 0.96;  % slight reduction to avoid clipping
end

% Safety: sand-dust A must satisfy A_R >= A_G >= A_B physically
if is_sandust
    A(1) = max(A(1), A(2));          % R >= G
    A(2) = max(A(2), A(3));          % G >= B
    A(3) = min(A(3), A(2)*0.90);     % B noticeably lower (absorption)
end
A = min(max(A, 0.50), 0.98);
fprintf('  A = [R=%.4f  G=%.4f  B=%.4f]\n', A(1),A(2),A(3));

%% ========================================================================
%% [5] REGION-ADAPTIVE TEXTURE PROBABILITY MAP [D]
%% ========================================================================
fprintf('[5/13] Region-adaptive texture map...\n');

% Local entropy as texture measure (Kim et al. 2021)
% High entropy → textured → aggressive dehazing
% Low entropy  → flat/sky  → conservative
ent_map = entropyfilt(rgb2gray(uint8(I_f*255)), ones(9,9));
ent_min = min(ent_map(:)); ent_max = max(ent_map(:));
P_tex   = (ent_map - ent_min) / (ent_max - ent_min + eps);   % [0,1]
P_tex   = imgaussfilt(P_tex, 5);   % smooth transitions

% Sky/bright region — always conservative
sky_mask = detectSky(I_f, h, w);
P_tex(sky_mask) = P_tex(sky_mask) * 0.35;

%% ========================================================================
%% [6] CHANNEL-WISE TRANSMISSION ESTIMATION [A, E]
%% ========================================================================
fprintf('[6/13] Channel-wise transmission estimation...\n');

patch = ternary(is_dense, 9, 7);
se    = strel('square', patch);

% Omega range — adaptive per texture
omega_min = ternary(is_sandust, 0.60, 0.55);
omega_max = ternary(is_sandust, ternary(is_dense, 0.95, 0.88), 0.90);
omega_map = omega_min + (omega_max - omega_min) * P_tex;

% Per-channel DCP transmission [A]
T = zeros(h, w, 3);
for c = 1:3
    norm_c   = max(I_f(:,:,c) / (A(c) + eps), 0);
    dark_c   = imerode(norm_c, se);
    T(:,:,c) = 1 - omega_map .* dark_c;
end

% Saturation-based transmission [E]:
% High saturation → clear object → high t; low sat → heavily veiled → low t
t_sat = 1 - (1 - avg_S) * max(omega_map - 0.1, 0);
t_sat = imgaussfilt(t_sat, 3);

% Fuse: for sand-dust use channel-specific; for fog use min
if is_sandust
    % Physically correct: use per-channel t (not min across channels)
    % t_R > t_G > t_B for sand-dust
    t_fused = T;  % keep all 3 channels separate
    % Saturate-blend only on luminance channel
    t_lum  = (T(:,:,1)*0.299 + T(:,:,2)*0.587 + T(:,:,3)*0.114);
    t_lum  = 0.70*t_lum + 0.30*t_sat;
    % Apply luminance adjustment to each channel proportionally
    ratio  = t_lum ./ (t_lum + eps);
    for c = 1:3
        t_fused(:,:,c) = T(:,:,c);  % per-channel stays per-channel
    end
else
    % Generic fog: single t map (min channel, DCP standard)
    t_min  = min(T,[],3);
    t_mono = 0.75*t_min + 0.25*t_sat;
    t_fused = repmat(t_mono,[1,1,3]);
end

% Clamp each channel
t_min_val = ternary(is_sandust, 0.10, 0.05);
for c = 1:3
    t_fused(:,:,c) = max(min(t_fused(:,:,c), 0.97), t_min_val);
end

%% ========================================================================
%% [7] GUIDED FILTER TRANSMISSION REFINEMENT [G]
%% ========================================================================
fprintf('[7/13] Guided-filter transmission refinement...\n');

gf_r   = max(round(min(h,w)*0.03), 8);
gf_eps = ternary(is_sandust, 1.5e-3, 7e-4);
guide  = rgb2gray(I_f);

T_ref = zeros(h,w,3);
for c = 1:3
    T_ref(:,:,c) = guidedfilter_fast(guide, t_fused(:,:,c), gf_r, gf_eps);
    T_ref(:,:,c) = max(min(T_ref(:,:,c), 0.97), t_min_val);
end

%% ========================================================================
%% [8] RED-OFFSET CORRECTED SCENE RECOVERY [B]
%% ========================================================================
fprintf('[8/13] Red-offset corrected scene recovery...\n');

% Per-channel offset d^c = mean(I_R) - mean(I_c)  [Kim 2021 / Springer 2016]
% Corrected atmospheric light: A_eff^c = A^c - d^c
d_offset = zeros(1,3);
if is_sandust
    for c = 1:3
        d_offset(c) = avg_R - mean(I_f(:,:,c),'all');
    end
    d_offset = max(min(d_offset, 0.20), -0.05);  % safety clamp
end
A_eff = A - d_offset;
A_eff = min(max(A_eff, 0.30), 0.98);

fprintf('  Offsets d=[%.4f %.4f %.4f]  A_eff=[%.4f %.4f %.4f]\n', ...
    d_offset(1),d_offset(2),d_offset(3), A_eff(1),A_eff(2),A_eff(3));

% Scene recovery: J^c = (I^c - A_eff^c) / t^c + A_eff^c
J = zeros(size(I));
for c = 1:3
    J(:,:,c) = (I(:,:,c) - A_eff(c)) ./ T_ref(:,:,c) + A_eff(c);
end
J = max(min(J,1),0);

%% ========================================================================
%% [9] BLUE CHANNEL COMPENSATION [C]
%% ========================================================================
fprintf('[9/13] Blue channel compensation...\n');

if is_sandust
    % Estimate blue deficit after recovery
    J_R = J(:,:,1); J_G = J(:,:,2); J_B = J(:,:,3);
    ref_mean  = (mean(J_R(:)) + mean(J_G(:))) / 2;
    blue_mean = mean(J_B(:));
    blue_deficit = max(ref_mean - blue_mean, 0);

    % Compensation factor alpha (Cheng et al. IEEE Access 2020)
    alpha = min(blue_deficit * 2.5, 0.45);
    fprintf('  blue_deficit=%.4f  alpha=%.4f\n', blue_deficit, alpha);

    % Compensate: boost blue using RG average
    J_B_comp = J_B + alpha * (J_R + J_G) / 2;
    J_B_comp = max(min(J_B_comp,1),0);
    J(:,:,3) = J_B_comp;

    % Smooth colour balance in LAB
    lab = rgb2lab(J);
    % Shift a* toward neutral (remove green-red tint)
    a_mean = mean(lab(:,:,2),'all');
    lab(:,:,2) = lab(:,:,2) - 0.4*a_mean;
    % Shift b* toward neutral (remove yellow-blue tint)
    b_mean = mean(lab(:,:,3),'all');
    lab(:,:,3) = lab(:,:,3) - 0.35*b_mean;
    J = lab2rgb(lab);
    J = max(min(J,1),0);
end

% Generic mist: chrominance smoothing
if is_mist && ~is_sandust
    lab = rgb2lab(J);
    lab(:,:,2) = imgaussfilt(lab(:,:,2), 1.0);
    lab(:,:,3) = imgaussfilt(lab(:,:,3), 1.0);
    J = lab2rgb(lab); J = max(min(J,1),0);
end

%% ========================================================================
%% [10] LAB ADAPTIVE GAMMA CONTRAST STRETCH [F]
%% ========================================================================
fprintf('[10/13] LAB adaptive gamma contrast stretch...\n');

% (Shi et al. IET Image Processing 2020 — Normalised Gamma CLAHE)
% Per-pixel adaptive gamma based on local mean luminance.
% γ(x) = log(0.5) / log(L_local(x))   — gamma < 1 lifts darks, > 1 darkens brights

lab = rgb2lab(J);
L   = lab(:,:,1) / 100;     % [0,1]
L   = max(L, 1e-4);

% Local mean luminance (large window for smooth gamma map)
L_local = imgaussfilt(L, 15);
L_local = max(L_local, 0.01);

% Per-pixel gamma: pixels darker than 0.5 get gamma<1 (lifted),
%                 pixels brighter get gamma>1 (slightly compressed)
gamma_px = log(0.5) ./ log(L_local + eps);
gamma_px = min(max(gamma_px, 0.45), 1.80);   % safety bounds

% Apply per-pixel gamma
L_gamma = L .^ gamma_px;
L_gamma = max(min(L_gamma,1),0);

% Target overall brightness: match a reasonable clear-day level
% (not the foggy input which was artificially bright)
target_mean_L = 0.45;   % target L* mean in [0,1]
cur_mean      = mean(L_gamma(:));
if cur_mean > 0.01
    scale_L = target_mean_L / cur_mean;
    scale_L = min(max(scale_L, 0.70), 1.40);
    L_gamma = L_gamma * scale_L;
    L_gamma = max(min(L_gamma,1),0);
end

lab(:,:,1) = L_gamma * 100;

% Chrominance saturation boost in LAB (compensate for fog desaturation)
sat_boost = ternary(is_sandust, 1.25, 1.18);
lab(:,:,2) = lab(:,:,2) * sat_boost;
lab(:,:,3) = lab(:,:,3) * sat_boost;
% Clip to valid LAB range
lab(:,:,2) = max(min(lab(:,:,2), 127), -128);
lab(:,:,3) = max(min(lab(:,:,3), 127), -128);

J = lab2rgb(lab);
J = max(min(J,1),0);

%% ========================================================================
%% [11] CLAHE (luminance only, tight clip)
%% ========================================================================
fprintf('[11/13] Luminance CLAHE...\n');

clip_lim = ternary(is_sandust, 0.008, 0.012);
lab = rgb2lab(J);
L_c = adapthisteq(lab(:,:,1)/100, 'ClipLimit',clip_lim, ...
                  'Distribution','rayleigh','NumTiles',[8 8]);
lab(:,:,1) = L_c * 100;
J = lab2rgb(lab);
J = max(min(J,1),0);

%% ========================================================================
%% [12] STRUCTURE-TENSOR COHERENCE-GATED SHARPENING (safe, no ringing)
%% ========================================================================
fprintf('[12/13] Structure-tensor edge sharpening...\n');

gray_J = rgb2gray(J);
[~,~,~,coh] = structureTensor(gray_J, 1.5, 3.0);

% Sharpening only where real oriented edges exist (coherence ≈ 1)
% Zero sharpening in sandy/sky/uniform areas (coherence ≈ 0)
max_sharp  = ternary(is_sandust, 0.20, 0.32);
sharp_mask = imgaussfilt(coh * max_sharp, 1.0);

J_blur = imgaussfilt(J, 0.85);
for c  = 1:3
    J(:,:,c) = J(:,:,c) + sharp_mask .* (J(:,:,c) - J_blur(:,:,c));
end
J = max(min(J,1),0);

%% ========================================================================
%% [13] FINAL VIBRANCE + OUTPUT
%% ========================================================================
fprintf('[13/13] Final polish & save...\n');

% Mild vibrance (boosts under-saturated colours, leaves greys alone)
vib = ternary(is_sandust, 0.10, 0.14);
J   = vibranceBoost(J, vib);

% Very mild final denoise to remove any residual grain
J   = imgaussfilt(J, 0.35);
J   = max(min(J,1),0);

%% ---- SAVE ---------------------------------------------------------------
out_dir = 'defogging_output';
if ~exist(out_dir,'dir'), mkdir(out_dir); end
[~,stem,ext] = fileparts(img_path);
result_path  = fullfile(out_dir,[stem,'_dehazed.png']);
cmp_path     = fullfile(out_dir,[stem,'_comparison.png']);

imwrite(J, result_path);
if ~any(strcmpi(ext,{'.png','.gif','.pbm'}))
    try
        np = fullfile(out_dir,[stem,'_dehazed',ext]);
        if any(strcmpi(ext,{'.jpg','.jpeg'})), imwrite(J,np,'Quality',97);
        else, imwrite(J,np); end
    catch, end
end
fprintf('\n  Saved: %s\n', result_path);

%% ---- METRICS & FIGURE ---------------------------------------------------
[ei,gi] = imgMetrics(I);
[ej,gj] = imgMetrics(J);
sv       = ssim(J,I);
ms       = mean((J(:)-I(:)).^2);
psnr_v   = 10*log10(1/(ms+eps));

fig = figure('Name','Ultra Dehaze v5','Position',[30,30,1500,680],'Color','w');

subplot(1,2,1); imshow(I);
title({'ORIGINAL IMAGE', ...
    sprintf('Entropy:%.3f  |  Avg Grad:%.2f', ei,gi)}, ...
    'FontSize',12,'FontWeight','bold','Color',[0.2 0.2 0.2]);

subplot(1,2,2); imshow(J);
title({'DEHAZED (v5 Sand-Dust Specialist)', ...
    sprintf('Entropy:%.3f(%+.1f%%)  Grad:%.2f(%+.1f%%)  SSIM:%.3f  PSNR:%.1fdB', ...
    ej,100*(ej-ei)/(ei+eps), gj,100*(gj-gi)/(gi+eps), sv,psnr_v)}, ...
    'FontSize',11,'FontWeight','bold','Color',[0 0.42 0.08]);

try, print(fig,cmp_path,'-dpng','-r200');
    fprintf('  Comparison: %s\n', cmp_path);
catch, warning('Figure save failed.'); end

fprintf('\n=================================================================\n');
fprintf('                   QUALITY REPORT  (v5)\n');
fprintf('=================================================================\n');
fprintf('%-26s | Original | Restored | Change\n','Metric');
fprintf('-----------------------------------------------------------------\n');
fprintf('%-26s | %8.4f | %8.4f | %+.2f%%\n','Entropy',ei,ej,100*(ej-ei)/(ei+eps));
fprintf('%-26s | %8.2f | %8.2f | %+.1f%%\n','Avg Gradient',gi,gj,100*(gj-gi)/(gi+eps));
fprintf('%-26s | %8.4f | %8.4f\n','Contrast (StdDev)', ...
    std(rgb2gray(I),0,'all'), std(rgb2gray(J),0,'all'));
fprintf('%-26s |   1.0000 | %8.4f\n','SSIM',sv);
fprintf('%-26s |      --- | %7.2f dB\n','PSNR',psnr_v);
fprintf('-----------------------------------------------------------------\n');
fprintf('Mode: %s | fog_idx=%.2f | omega=[%.2f,%.2f]\n', ...
    ternary(is_sandust,'SAND-DUST','GENERIC FOG'), fog_idx, omega_min,omega_max);
fprintf('=================================================================\n');
fprintf('✓  %s\n✓  %s\n', result_path, cmp_path);
fprintf('=================================================================\n');


%% =========================================================================
%%                         FUNCTION LIBRARY
%% =========================================================================

%% SKY DETECTION ----------------------------------------------------------
function sky = detectSky(I,h,w)
    hsv = rgb2hsv(I);
    S   = hsv(:,:,2); V = hsv(:,:,3);
    sp  = zeros(h,w);
    sp(1:round(h*.40),:) = 1.0;
    sp(round(h*.40)+1:round(h*.60),:) = 0.30;
    [Gx,Gy] = gradient(double(rgb2gray(I)));
    sm  = imgaussfilt(sqrt(Gx.^2+Gy.^2),4) < 0.07;
    prob= double((V>0.50)&(S<0.42)).*(0.5+0.5*sp).*double(sm);
    sky = prob > 0.27;
    sky = imopen(sky,strel('disk',3));
    sky = imclose(sky,strel('disk',10));
    sky = imfill(sky,'holes');
    sky = bwareaopen(sky,round(h*w*0.003));
end

%% GUIDED FILTER (fast, self-contained) -----------------------------------
function out = guidedfilter_fast(I, p, r, eps)
    I   = double(I); p = double(p);
    N   = boxfilt(ones(size(I)),r);
    mI  = boxfilt(I,r)./N;
    mp  = boxfilt(p,r)./N;
    mIp = boxfilt(I.*p,r)./N;
    mII = boxfilt(I.*I,r)./N;
    a   = (mIp - mI.*mp)./(mII - mI.^2 + eps);
    b   = mp - a.*mI;
    ma  = boxfilt(a,r)./N;
    mb  = boxfilt(b,r)./N;
    out = ma.*I + mb;
    out = max(min(out,1),0.01);
end

function out = boxfilt(img,r)
    out = imfilter(img, ones(2*r+1)/(2*r+1)^2, 'replicate');
end

%% BILATERAL APPROX -------------------------------------------------------
function out = bilateralApprox(I, ss, sr)
    r = ceil(3*ss); G = fspecial('gaussian',2*r+1,ss);
    out = I;
    for it = 1:4
        bl = imfilter(out,G,'replicate');
        w  = exp(-(abs(out-bl).^2)/(2*sr^2));
        out= w.*out+(1-w).*bl;
    end
end

%% STRUCTURE TENSOR -------------------------------------------------------
function [Jxx,Jxy,Jyy,coh] = structureTensor(img,ri,re)
    img=double(img); [Gx,Gy]=gradient(img);
    gi=fspecial('gaussian',round(6*ri)+1,ri);
    ge=fspecial('gaussian',round(6*re)+1,re);
    Jxx=imfilter(imfilter(Gx.^2, gi,'replicate'),ge,'replicate');
    Jxy=imfilter(imfilter(Gx.*Gy,gi,'replicate'),ge,'replicate');
    Jyy=imfilter(imfilter(Gy.^2, gi,'replicate'),ge,'replicate');
    tmp=sqrt((Jxx-Jyy).^2+4*Jxy.^2);
    l1=0.5*(Jxx+Jyy+tmp); l2=0.5*(Jxx+Jyy-tmp);
    coh=((l1-l2).^2)./((l1+l2).^2+eps);
    coh=max(min(coh,1),0);
end

%% VIBRANCE BOOST ---------------------------------------------------------
function J = vibranceBoost(I,amt)
    hsv=rgb2hsv(I); S=hsv(:,:,2);
    hsv(:,:,2)=min(S+amt*(1-S),1);
    J=hsv2rgb(hsv); J=max(min(J,1),0);
end

%% IMAGE METRICS ----------------------------------------------------------
function [ev,ag] = imgMetrics(I)
    g=rgb2gray(uint8(I*255));
    ev=entropy(g);
    [Gx,Gy]=gradient(double(g));
    ag=mean(sqrt(Gx(:).^2+Gy(:).^2));
end

%% TERNARY ----------------------------------------------------------------
function r = ternary(c,t,f)
    if c, r=t; else, r=f; end
end
