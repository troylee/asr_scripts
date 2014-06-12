function noisyFBank=MaskedFBank_htk(noisywav, mask)
% 
% This function extract 24D log Mel FBanks from WAV
% format wave file. The signal is masked by Ideal Binary Mask(IBM) in the
% power spectrum domain before FBank feature extraction. Requires stereo
% data.
%
% The extracted features are exactly the same as using HTK. 
% The configuration parameters are based on Aurora4. 
%
% No delta and accelerate parameters are computed
% 
%
%
% Inputs:
%   noisywav - noisy speech signal, WAV format
%   mask     - mask in power spectrum domain
%
% Outputs:
%   noisyFBank - FBank features of the masked noisy signal
%
% Jun.19, 2013
%

if nargin < 2
    disp('Insufficient input arguments!');
    return;
end

%% for wav format, needs to read the native integer data, not the normalized value
[noisy_s, noisy_fs] = wavread(noisywav,'native');
noisy_s = double(noisy_s);
% common fs
fs = noisy_fs;


%% %%%%%%%   Common parameters
% window length is 25.0ms
windowsize = fix(0.025 * fs);

% frame rate is 10ms
targetrate = round(0.01 * fs);

% source rate, number of samples in 100ns (1e-7s)
sourcerate = 625.0;

% pre-emphasise coefficient
preEmph = 0.97;

% FFT length
fftlen = pow2(nextpow2(windowsize));

% number of FBank channels
numChans = 24;

%% %%%%%%%  split the samples into overlapping frames
numsam = length(noisy_s(:)); % the same to clean_s
numfrm = fix((numsam-windowsize+targetrate)/targetrate);
indf = targetrate * (0:(numfrm-1)).';
inds = (1:windowsize);
% the frmdata is organized that each row is a frame.
noisy_dataFrm = noisy_s(indf(:,ones(1,windowsize))+inds(ones(numfrm,1),:));

%% %%%%%%%  Pre-Processing
% ZeroMeanSource, done per frame
noisy_frameMean = mean(noisy_dataFrm, 2);
noisy_dataFrm = noisy_dataFrm - noisy_frameMean(:, ones(1, windowsize));

% pre-emphasise
preEmphmat = eye(windowsize);
preEmphmat(1,1) = 1 - preEmph;
for i=2:windowsize,
	preEmphmat(i-1,i) = -preEmph;
end
noisy_dataFrm = noisy_dataFrm * preEmphmat;

% hamming window
hamWin = 0.54 - 0.46 * cos(2*pi*(0:windowsize-1)/(windowsize-1));
for fid=1:numfrm,
	noisy_dataFrm(fid,:) = noisy_dataFrm(fid,:).*hamWin;
end

%% Computing Spectrum Features

% FFT
Nby2=fftlen/2;
noisy_dataFreq=rfft(noisy_dataFrm, fftlen, 2);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% apply the mask on the spectrum
% 
% apply the mask (mask is only applied to magnitude, phase is the same)
noisy_dataFreq = abs(noisy_dataFreq) .* mask .* exp(j*angle(noisy_dataFreq));

%% Computing FBank features
% Frequency resolution
fres=1.0e7/(sourcerate*fftlen*700.0);

% Setting up the constants for FBank computation
% Default low and high pass cut offs
klo=2;
khi=Nby2;
mlo=0;
mhi=Mel(Nby2+1,fres);
ms=mhi-mlo;

% FBank center frequency, by deafult no warp factor
maxChan=numChans+1;
cf=(1:maxChan)*ms/maxChan+mlo;

% create loChan map
loChan=zeros(1,Nby2);
chan=1;
for k=1:Nby2,
	melk=Mel(k,fres);
	%disp(['k=' num2str(k) ' melk=' num2str(melk)]); 
	if (k<klo) || (k>khi),
		loChan(k)=-1;
	else
		while (chan<=maxChan) && (cf(chan)<melk),
			chan=chan+1;
		end
		%disp(chan);
		loChan(k)=chan-1;
	end
end

% create loWt 
loWt=zeros(1,Nby2);
for k=1:Nby2,
	chan=loChan(k);
	if k<klo || k>khi,
		loWt(k)=0.0;
	else
		if chan>0,
			loWt(k)=((cf(chan+1)-Mel(k,fres))/(cf(chan+1)-cf(chan)));
		else
			loWt(k)=(cf(1)-Mel(k,fres))/(cf(1)-mlo);
		end
	end
end

% compute fbank vectors
noisyFBank=zeros(numfrm,numChans);

melfloor=1.0;

for fid=1:numfrm,
	for k=klo:khi,
		% by default use magnitude not the power
		ek=sqrt(noisy_dataFreq(fid,k).*conj(noisy_dataFreq(fid,k)));
		
		bin=loChan(k);
		t1=loWt(k)*ek;
		if bin>0,
			noisyFBank(fid,bin)=noisyFBank(fid,bin)+t1;
		end
		if bin<numChans,
			noisyFBank(fid,bin+1)=noisyFBank(fid,bin+1)+ek-t1;
		end
	end
	
	% taking log
	for bin=1:numChans,
		t1=noisyFBank(fid,bin);
		if t1<melfloor,
			t1=melfloor;
		end
		noisyFBank(fid,bin)=log(t1);
	end
	
end
