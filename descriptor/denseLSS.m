function des = denseLSS(img,desc_rad,nrad,nang);


parms.patch_size=3;
parms.desc_rad=desc_rad;
parms.nrad=nrad;
parms.nang=nang;
parms.var_noise=3000;
parms.saliency_thresh = 1;
%parms.saliency_thresh = 0.7;
parms.homogeneity_thresh=1;
%parms.homogeneity_thresh=0.7;
parms.snn_thresh=1; % I usually disable saliency checking
%parms.snn_thresh=0.85;
%parms.nChannels=size(i,3);
des_num = parms.nrad*parms.nang;
%des_num = (2*desc_rad+1)*(2*desc_rad+1);
margin = parms.desc_rad + (parms.patch_size-1)/2;
img = padarray(img,[margin,margin],'symmetric');
img = double(img(:,:,1));
[h,w] = size(img);


des_width = w-2*margin; % the width of descriptor
des_height = h-2*margin;% the height of descriptor

destmp = zeros(des_height,des_width,des_num);
des = zeros(h,w,des_num);

%[resp, draw_coords, salient_coords, homogeneous_coords, snn_coords] = mexCalcSsdescs(img, parms);
[resp, draw_coords, salient_coords, homogeneous_coords, snn_coords] = mexCalcSsdescs1(img, parms);
%[resp, draw_coords, salient_coords, homogeneous_coords, snn_coords] = mexCalcSsdescs_mean(img, parms);
%[resp, draw_coords, salient_coords, homogeneous_coords, snn_coords] = mexCalcSSD(img, parms);
%[resp, draw_coords, salient_coords, homogeneous_coords, snn_coords] = mexCalcSSDslow(img, parms);

resp = resp';
temp = reshape(resp,[des_width,des_height,des_num]);
temp1 = permute(temp,[2 1 3]);

destmp = temp1;

%des(margin:h-margin-1,margin:w-margin-1,:) = destmp;
%des(margin+1:h-margin,margin+1:w-margin,:) = destmp;
%des = single(des);
des =single(destmp);
des = permute(des, [3 1 2]);

