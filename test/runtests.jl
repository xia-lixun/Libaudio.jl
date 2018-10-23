import Libaudio
using Test


function wav2pcm_test()
    foo = joinpath(Libaudio.modulepath(Libaudio), "test/acqua_ieee_male_250ms_10450ms.wav")
    Libaudio.wav2pcm(foo, foo[1:end-4] * "-16.pcm", 16)
    Libaudio.wav2pcm(foo, foo[1:end-4] * "-24.pcm", 24)
    Libaudio.wav2pcm(foo, foo[1:end-4] * "-32.pcm", 32)
    nothing
end
let y = wav2pcm_test()
    # use Adobe Audition to check the results
end


function libwav_test()
    filename = joinpath(Libaudio.modulepath(Libaudio), "test/acqua_ieee_male_250ms_10450ms.wav")
    Libaudio.wavmeta(filename)
    r,fs = Libaudio.wavread(filename)
    # m = Libaudio.wavinfo(filename)
    # x = zeros(Float32, m[1] * m[2])
    # Libaudio.wavread!(filename, x)
    # y = permutedims(reshape(x, m[2], m[1]))
    ys,sr,bps = Libaudio.wavread(filename)
    yd,sr,bps = Libaudio.wavread(filename, Float64)

    Libaudio.wavwrite(filename[1:end-4] * "_s32.wav", ys, Int(fs), 32)
    Libaudio.wavwrite(filename[1:end-4] * "_s24.wav", ys, Int(fs), 24)
    Libaudio.wavwrite(filename[1:end-4] * "_s16.wav", ys, Int(fs), 16)
    Libaudio.wavwrite(filename[1:end-4] * "_d32.wav", yd, Int(fs), 32)
    Libaudio.wavwrite(filename[1:end-4] * "_d24.wav", yd, Int(fs), 24)
    Libaudio.wavwrite(filename[1:end-4] * "_d16.wav", yd, Int(fs), 16)
    (r,ys,yd)
end
let (r,ys,yd) = libwav_test()
    @test isapprox(r, ys, atol=1e-7)
    @test isapprox(r, yd, atol=1e-7)
    @info "==== (0) libwav_test ===="
end


function wavio_test()
    path = Libaudio.modulepath(Libaudio)
    x,fs,nb = Libaudio.wavread(joinpath(path,"test/acqua_ieee_male_250ms_10450ms.wav"))
    Libaudio.wavwrite(joinpath(path,"test/foo24.wav"), x, Int(fs), 24)
    Libaudio.wavwrite(joinpath(path,"test/foo16.wav"), x, Int(fs), 16)
    Libaudio.wavwrite(joinpath(path,"test/foo32.wav"), x, Int(fs), 32)

    x16,fs,nb = Libaudio.wavread(joinpath(path,"test/foo16.wav"))
    x24,fs,nb = Libaudio.wavread(joinpath(path,"test/foo24.wav"))
    x32,fs,nb = Libaudio.wavread(joinpath(path,"test/foo32.wav"))
    x, x16, x24, x32
end
let (x, x16, x24, x32) = wavio_test()
    @test isapprox(x16, x, atol=1)
    @test isapprox(x24, x, atol=1e-4)
    @test isapprox(x32, x, atol=1e-7)
    @info "==== (1) wavio_test ===="
end


function wavalid_test()
    path = Libaudio.modulepath(Libaudio)
    x = 0.1 * randn(192000,8)
    y = convert(Matrix{Float32},x)
    Libaudio.wavwrite(joinpath(path,"test/valid_d2f.wav"), x, 48000, 32)
    Libaudio.wavwrite(joinpath(path,"test/valid_f2f.wav"), y, 48000, 32)

    xd,fs,nb = Libaudio.wavread(joinpath(path,"test/valid_d2f.wav"), Float64)
    yf1,fs,nb = Libaudio.wavread(joinpath(path,"test/valid_f2f.wav"), Float32)
    yf2,fs,nb = Libaudio.wavread(joinpath(path,"test/valid_f2f.wav"))
    x,xd,y,yf1,yf2
end
let (x,xd,y,yf1,yf2) = wavalid_test()
    @test y == yf1
    @test y == yf2
    @test isapprox(x, xd, atol=1e-5)
    @info "==== (2) wavalid_test ===="
end



function weighting_a_truth_48000()
    AWEIGHT_48kHz_BA = [0.234301792299513 -0.468603584599025 -0.234301792299515 0.937207169198055 -0.234301792299515 -0.468603584599025 0.234301792299512;
                            1.000000000000000 -4.113043408775872 6.553121752655049 -4.990849294163383 1.785737302937575 -0.246190595319488 0.011224250033231]'                            
end

function weighting_a_truth_16000()
    AWEIGHT_16kHz_BA = [0.531489829823557 -1.062979659647115 -0.531489829823556 2.125959319294230 -0.531489829823558 -1.062979659647116 0.531489829823559;
    1.000000000000000 -2.867832572992163  2.221144410202311 0.455268334788664 -0.983386863616284 0.055929941424134 0.118878103828561]'
end

let (b, a) = Libaudio.weighting_a(48000), truth = weighting_a_truth_48000()
    @test isapprox(b, truth[:,1], atol = 1e-14)
    @test isapprox(a, truth[:,2], atol = 1e-14)
    @info "==== (3.1) A weighting ===="
end
let (b, a) = Libaudio.weighting_a(16000), truth = weighting_a_truth_16000()
    @test isapprox(b, truth[:,1], atol = 1e-14)
    @test isapprox(a, truth[:,2], atol = 1e-14)
    @info "==== (3.2) A weighting ===="
end



function conv_truth()
    convert(Vector{Float64}, Libaudio.conv_naive(collect(1.0:1.0:4.0), collect(4.0:-1.0:-4.0)))
end
let y = Libaudio.conv(collect(1.0:1.0:4.0), collect(4.0:-1.0:-4.0)), truth = conv_truth()
    @test isapprox(y, truth, atol = 1e-14)
    @info "==== (4) conv ===="
end


function xcorr_truth()
    # octave
    y = [-4.0, -11.0, -20.0, -30.0, -20.0, -10.0, -0.0, 10.0, 20.0, 25.0, 24.0, 16.0, 0.0, -0.0, -0.0, 0.0, 0.0]
end
let y = Libaudio.xcorr(collect(1.0:1.0:4.0), collect(4.0:-1.0:-4.0)), truth = xcorr_truth()
    @test isapprox(y, truth, atol = 1e-14)
    @info "==== (5) xcorr ===="
end


function xcorrcoeff_test()
    x = Libaudio.xcorrcoeff(collect(1.0:1.0:4.0), collect(4.0:-1.0:-4.0))
    y = Libaudio.xcorrcoeff_threaded(collect(1.0:1.0:4.0), collect(4.0:-1.0:-4.0))
    (x,y)
end
let (x,y) = xcorrcoeff_test()
    @test x == y
    @info "==== (6) xcorrcoeff_test ===="
end


function filt_truth()
    y = [3.150000000000000e+01, -1.675000000000000e+01, 5.125000000000000e+00, 5.031250000000000e+01, -4.271875000000000e+01, 3.453125000000000e+00, 1.377578125000000e+02, -1.690429687500000e+02]
    [y y y y]
end
let b = [7,6,5], a = [2,3,4], x = Float32[9,1,5,2,7,4,8,3], truth = filt_truth()
    y = Libaudio.filt(b,a,[x x x x])
    @test isapprox(y, truth, atol = 1e-17)
    @info "==== (7) filt ===="
end




function hamming_truth_periodic()
    w = [8.000000000000002e-02, 2.147308806541882e-01, 5.400000000000000e-01, 8.652691193458119e-01, 1.000000000000000e+00, 8.652691193458120e-01, 5.400000000000001e-01, 2.147308806541882e-01]
end
function hamming_truth_symmetric()
    w = [8.000000000000002e-02, 2.531946911449826e-01, 6.423596296199047e-01, 9.544456792351128e-01, 9.544456792351128e-01, 6.423596296199048e-01, 2.531946911449827e-01, 8.000000000000002e-02]
end
function hann_truth_periodic()
    w = [0.000000000000000e+00, 1.464466094067262e-01, 4.999999999999999e-01, 8.535533905932737e-01, 1.000000000000000e+00, 8.535533905932738e-01, 5.000000000000001e-01, 1.464466094067263e-01]
end
function hann_truth_symmetric()
    w = [0.000000000000000e+00, 1.882550990706332e-01, 6.112604669781572e-01, 9.504844339512095e-01, 9.504844339512095e-01, 6.112604669781573e-01, 1.882550990706333e-01, 0.000000000000000e+00]
end

let w = Libaudio.hamming(8), truth = hamming_truth_symmetric()
    @test isapprox(w, truth, atol = 1e-15)
    @info "==== (8.1) hamming symmetric ===="
end
let w = Libaudio.hamming(8, "periodic"), truth = hamming_truth_periodic()
    @test isapprox(w, truth, atol = 1e-15)
    @info "==== (8.2) hamming periodic ===="
end
let w = Libaudio.hann(8), truth = hann_truth_symmetric()
    @test isapprox(w, truth, atol = 1e-15)
    @info "==== (8.3) hann symmetric ===="
end
let w = Libaudio.hann(8, "periodic"), truth = hann_truth_periodic()
    @test isapprox(w, truth, atol = 1e-15)
    @info "==== (8.4) hann periodic ===="
end




function zeropend_truth()
    ff = ([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
    tf = (Float64[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 4)
    ft = (Float32[1, 2, 3, 4, 5, 6, 7, 8, 9, 0], 4)
    tt = (ComplexF64[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0], 5)
    (ff, tf, ft, tt)
end
let (ff, tf, ft, tt) = zeropend_truth()
    @test ff == Libaudio.zeropend(collect(1:9), Libaudio.WindowFrame(4,2), false, false)
    @test tf == Libaudio.zeropend(collect(1.:9.), Libaudio.WindowFrame(4,2), true, false)
    @test ft == Libaudio.zeropend(collect(1.f0:9.f0), Libaudio.WindowFrame(4,2), false, true)
    @test tt == Libaudio.zeropend(ComplexF64.(1:9), Libaudio.WindowFrame(4,2), true, true)
    @info "==== (9) zeropend ===="
end



function buffer_truth()
    ff = (Complex{Float64}[1.0+0.0im 3.0+0.0im 5.0+0.0im; 2.0+0.0im 4.0+0.0im 6.0+0.0im; 3.0+0.0im 5.0+0.0im 7.0+0.0im; 4.0+0.0im 6.0+0.0im 8.0+0.0im], 8)
    tf = ([0.0 1.0 3.0 5.0; 0.0 2.0 4.0 6.0; 1.0 3.0 5.0 7.0; 2.0 4.0 6.0 8.0], 10)
    ft = ([1 3 5 7; 2 4 6 8; 3 5 7 9; 4 6 8 0], 10)
    tt = (Float32[0 1 3 5 7; 0 2 4 6 8; 1 3 5 7 9; 2 4 6 8 0], 12)
    (ff, tf, ft, tt)
end
let (ff, tf, ft, tt) = buffer_truth()
    @test ff == Libaudio.buffer(ComplexF64.(1.:9.), Libaudio.WindowFrame(4,2), false, false)
    @test tf == Libaudio.buffer(collect(1.:9.), Libaudio.WindowFrame(4,2), true, false)
    @test ft == Libaudio.buffer(collect(1:9), Libaudio.WindowFrame(4,2), false, true)
    @test tt == Libaudio.buffer(collect(1.f0:9.f0), Libaudio.WindowFrame(4,2), true, true)
    @info "==== (10) buffer ===="
end





function spectrogram_truth()
    # octave:
    # y = buffer(1:100, 16, 4)
    # fft(y)
    y = [
    7.800000000000000e+01 + 0.000000000000000e+00im   2.640000000000000e+02 + 0.000000000000000e+00im  4.560000000000000e+02 + 0.000000000000000e+00im  6.480000000000000e+02 + 0.000000000000000e+00im   8.400000000000000e+02 + 0.000000000000000e+00im   1.032000000000000e+03 + 0.000000000000000e+00im  1.224000000000000e+03 + 0.000000000000000e+00im   1.416000000000000e+03 + 0.000000000000000e+00im   7.720000000000000e+02 + 0.000000000000000e+00im;
    -2.445134153790879e+00 + 3.874624229109006e+01im  -8.000000000000000e+00 + 4.021871593700678e+01im  -8.000000000000000e+00 + 4.021871593700678e+01im  -8.000000000000000e+00 + 4.021871593700678e+01im  -8.000000000000000e+00 + 4.021871593700678e+01im  -8.000000000000000e+00 + 4.021871593700678e+01im  -8.000000000000000e+00 + 4.021871593700678e+01im  -8.000000000000000e+00 + 4.021871593700678e+01im   8.386292881545590e+01 - 4.876519307362072e+02im;
    -3.585786437626905e+00 + 1.689949493661167e+01im  -8.000000000000000e+00 + 1.931370849898476e+01im  -8.000000000000000e+00 + 1.931370849898476e+01im  -8.000000000000000e+00 + 1.931370849898476e+01im  -8.000000000000000e+00 + 1.931370849898476e+01im  -8.000000000000000e+00 + 1.931370849898476e+01im  -8.000000000000000e+00 + 1.931370849898476e+01im  -8.000000000000000e+00 + 1.931370849898476e+01im  -4.000000000000000e+00 + 9.656854249492380e+00im;
    -4.941739916456369e+00 + 9.417980255114790e+00im  -8.000000000000000e+00 + 1.197284610132391e+01im  -8.000000000000000e+00 + 1.197284610132391e+01im  -8.000000000000000e+00 + 1.197284610132391e+01im  -8.000000000000000e+00 + 1.197284610132391e+01im  -8.000000000000000e+00 + 1.197284610132391e+01im  -8.000000000000000e+00 + 1.197284610132391e+01im  -8.000000000000000e+00 + 1.197284610132391e+01im   9.538008559557824e+01 - 1.451707589785524e+02im;
    -6.000000000000000e+00 + 6.000000000000000e+00im  -8.000000000000000e+00 + 8.000000000000000e+00im  -8.000000000000000e+00 + 8.000000000000000e+00im  -8.000000000000000e+00 + 8.000000000000000e+00im  -8.000000000000000e+00 + 8.000000000000000e+00im  -8.000000000000000e+00 + 8.000000000000000e+00im  -8.000000000000000e+00 + 8.000000000000000e+00im  -8.000000000000000e+00 + 8.000000000000000e+00im  -4.000000000000000e+00 + 4.000000000000000e+00im;
    -6.472473645916727e+00 + 4.204776819518363e+00im  -8.000000000000000e+00 + 5.345429103354389e+00im  -8.000000000000000e+00 + 5.345429103354389e+00im  -8.000000000000000e+00 + 5.345429103354389e+00im  -8.000000000000000e+00 + 5.345429103354389e+00im  -8.000000000000000e+00 + 5.345429103354389e+00im  -8.000000000000000e+00 + 5.345429103354389e+00im  -8.000000000000000e+00 + 5.345429103354389e+00im   9.627676865391413e+01 - 6.481332787817199e+01im;
    -6.414213562373095e+00 + 2.899494936611665e+00im  -8.000000000000000e+00 + 3.313708498984761e+00im  -8.000000000000000e+00 + 3.313708498984761e+00im  -8.000000000000000e+00 + 3.313708498984761e+00im  -8.000000000000000e+00 + 3.313708498984761e+00im  -8.000000000000000e+00 + 3.313708498984761e+00im  -8.000000000000000e+00 + 3.313708498984761e+00im  -8.000000000000000e+00 + 3.313708498984761e+00im  -4.000000000000000e+00 + 1.656854249492381e+00im;
    -6.140652283836026e+00 + 1.533038855493633e+00im  -8.000000000000000e+00 + 1.591298939037266e+00im  -8.000000000000000e+00 + 1.591298939037266e+00im  -8.000000000000000e+00 + 1.591298939037266e+00im  -8.000000000000000e+00 + 1.591298939037266e+00im  -8.000000000000000e+00 + 1.591298939037266e+00im  -8.000000000000000e+00 + 1.591298939037266e+00im  -8.000000000000000e+00 + 1.591298939037266e+00im   9.648021693505173e+01 - 1.929449963582681e+01im;
    -6.000000000000000e+00 + 0.000000000000000e+00im  -8.000000000000000e+00 + 0.000000000000000e+00im  -8.000000000000000e+00 + 0.000000000000000e+00im  -8.000000000000000e+00 + 0.000000000000000e+00im  -8.000000000000000e+00 + 0.000000000000000e+00im  -8.000000000000000e+00 + 0.000000000000000e+00im  -8.000000000000000e+00 + 0.000000000000000e+00im  -8.000000000000000e+00 + 0.000000000000000e+00im  -4.000000000000000e+00 + 0.000000000000000e+00im   
    ]
end
let y = spectrogram_truth()
    x = Libaudio.spectrogram(collect(1.:100.), Libaudio.WindowFrame(16,12), true, true)
    @test isapprox(x[1], y, atol=1e-13)
    # note that we can: isapprox([1.0, 2.0], [1.0+eps(Float32), 2.0+eps(Float32)], atol = 1e-6) -> true
    @info "==== (11) spectrogram ===="
end



function powerspectrum_truth()
    # octave abs(fft(buffer(1:100, 16, 4))).^2 / 16
    y = [
        3.802500000000000e+02   4.356000000000000e+03   1.299600000000000e+04   2.624400000000000e+04   4.410000000000000e+04   6.656400000000000e+04  9.363600000000000e+04   1.253160000000000e+05   3.724900000000000e+04;
        9.420312329436815e+01   1.050965694763527e+02   1.050965694763527e+02   1.050965694763527e+02   1.050965694763527e+02   1.050965694763527e+02  1.050965694763527e+02   1.050965694763527e+02   1.530233727376606e+04;
        1.865317459305202e+01   2.731370849898476e+01   2.731370849898476e+01   2.731370849898476e+01   2.731370849898476e+01   2.731370849898476e+01  2.731370849898476e+01   2.731370849898476e+01   6.828427124746191e+00;
        7.069946592976890e+00   1.295931523537420e+01   1.295931523537420e+01   1.295931523537420e+01   1.295931523537420e+01   1.295931523537420e+01  1.295931523537420e+01   1.295931523537420e+01   1.885744374414300e+03;
        4.499999999999999e+00   8.000000000000002e+00   8.000000000000002e+00   8.000000000000002e+00   8.000000000000002e+00   8.000000000000002e+00  8.000000000000002e+00   8.000000000000002e+00   2.000000000000000e+00;
        3.723316449940346e+00   5.785850768686756e+00   5.785850768686756e+00   5.785850768686756e+00   5.785850768686756e+00   5.785850768686756e+00  5.785850768686756e+00   5.785850768686756e+00   8.418739783176704e+02;
        3.096825406947977e+00   4.686291501015240e+00   4.686291501015240e+00   4.686291501015240e+00   4.686291501015240e+00   4.686291501015240e+00  4.686291501015240e+00   4.686291501015240e+00   1.171572875253810e+00;
        2.503613662714615e+00   4.158264519586321e+00   4.158264519586321e+00   4.158264519586321e+00   4.158264519586321e+00   4.158264519586321e+00  4.158264519586321e+00   4.158264519586321e+00   6.050443735019726e+02;
        2.250000000000000e+00   4.000000000000000e+00   4.000000000000000e+00   4.000000000000000e+00   4.000000000000000e+00   4.000000000000000e+00  4.000000000000000e+00   4.000000000000000e+00   1.000000000000000e+00   
    ]
end
let y = powerspectrum_truth()
    x = Libaudio.powerspectrum(collect(1.0:100.0), Libaudio.WindowFrame(16,12), true, true)
    @test isapprox(x[1], y, atol=1e-11)
    @info "==== (12) powerspectrum ===="
end




"""
in this extreme case T=Float32 is insufficient in accuracy but I doubt for normal audio signals 
single precision would be enough?
"""
function extractsymbol_test(T)
    a = randn(T,8192)
    b = randn(T,8192)
    c = randn(T,8192)
    t = randn(T,512)
    lbs, pk, pkf, m = Libaudio.extractsymbol([zeros(T,1); T(0.05)*t; a; T(0.001)*t; b; T(0.1)*t; c; T(0.01)*t], t, 4, verbose=false, normcoeff=true)
    lbsx, pkx, pkfx, mx = Libaudio.extractsymbol([zeros(T,1); t; a; t; b; t; c; t], t, 4, verbose=false)
    # @info "extract symbol test: xcorrcoeff enabled" lbs pk pkf
    # @info "extract symbol test: xcorrcoeff disabled" lbsx pkx pkfx
    (lbs, pk, pkf, m, lbsx, pkx, pkfx, mx)
end
let (lbs, pk, pkf, m, lbsx, pkx, pkfx, mx) = extractsymbol_test(Float64)
    @test lbs[1] == 1 + 1
    @test lbs[2] == lbs[1]-1+(512+8192)+1
    @test lbs[3] == lbs[2]-1+(512+8192)+1
    @test lbs[4] == lbs[3]-1+(512+8192)+1
    @test lbsx[1] == 1 + 1
    @test lbsx[2] == lbsx[1]-1+(512+8192)+1
    @test lbsx[3] == lbsx[2]-1+(512+8192)+1
    @test lbsx[4] == lbsx[3]-1+(512+8192)+1
    @info "==== (13) extractsymbol_test ===="
end






function spl_test()
    calib250, fs = Libaudio.wavread(joinpath(Libaudio.modulepath(Libaudio), "test\\2018-06-22T15-37-20-913+42AA_114.0_105.4_26XX_12AA_0dB_UFX.wav"))
    calib1000, fs = Libaudio.wavread(joinpath(Libaudio.modulepath(Libaudio), "test\\2018-06-22T15-38-53-76+42AB_114.0__26XX_12AA_0dB_UFX.wav"))
    sp, fs = Libaudio.wavread(joinpath(Libaudio.modulepath(Libaudio), "test\\acqua_ieee_male_250ms_10450ms.wav"))

    aanw = Libaudio.spl(calib250[:,1], sp[:,1:1], sp[:,1], 1, Libaudio.WindowFrame(fs,16384,16384÷4), 0.3, 10.3, 100, 12000, 114.0)
    abnw = Libaudio.spl(calib1000[:,1], sp[:,1:1], sp[:,1], 1, Libaudio.WindowFrame(fs,16384,16384÷4), 0.3, 10.3, 100, 12000, 114.0)
    aaaw = Libaudio.spl(calib250[:,1], sp[:,2:2], sp[:,2], 1, Libaudio.WindowFrame(fs,16384,16384÷4), 0.3, 10.3, 100, 12000, 105.4, weighting="A")
    abaw = Libaudio.spl(calib1000[:,1], sp[:,2:2], sp[:,2], 1, Libaudio.WindowFrame(fs,16384,16384÷4), 0.3, 10.3, 100, 12000, 105.4, weighting="A") 
    return (aanw, abnw, aaaw, abaw)
end
let (aanw, abnw, aaaw, abaw) = spl_test()
    # @info "spl test results" aanw abnw aaaw abaw
    @test abs(aanw[1] - abnw[1]) < 0.5
    @test isapprox(aanw[1], 128.166, atol=1e-3)
    @test isapprox(abnw[1], 127.993, atol=1e-3)
    @test isapprox(aaaw[1], 125.182, atol=1e-3)
    @test isapprox(abaw[1], 116.33, atol=1e-3)
    @info "==== (14) spl_test ===="
end




function impulse_test()
    ndecay = 1000
    a = Libaudio.expsinesweep(100, 8000, 1, 16000)
    r = [zeros(BigFloat,17); a; zeros(BigFloat, ndecay-17)]
    r = [r r]
    f,h,d,m = Libaudio.impulse(a, ndecay, 100, 8000, 16000, r)
    
    u,q = findmax(view(d,:,1))
    v,p = findmax(view(m,:,1))
    (u,v,p,q)
end
let (u,v,p,q) = impulse_test()
    @test p-q == 17
    @test isapprox(u, v, atol=1e-15)
    @info "==== (15) impulse_test ===="
end



function symbol_test()
    fs = 48000
    x = randn(8192, 4)
    s = Libaudio.symbol_expsinesweep(800, 2000, 1, fs)
    se = Libaudio.encode_syncsymbol(0.5, s, 0.1, x, fs, 2, -6)
    Libaudio.decode_syncsymbol(se[:,2:2], s, 0.1, size(x,1)/fs, fs)
end
let loc = symbol_test()
    @test loc[1] == convert(Int, 0.5 * 48000 + length(Libaudio.symbol_expsinesweep(800, 2000, 1, 48000)) + 0.1 * 48000) + 1
    @info "==== (16) symbol_test ===="
end


function list_test()
    mp = Libaudio.modulepath(Libaudio)
    shallowfolders = Libaudio.list(mp)
    deepfolders = Libaudio.list(mp, recursive=true)
    shallowfiles = Libaudio.list(mp, ".jl")
    deepfiles = Libaudio.list(mp, ".jl", recursive=true)

    # @info shallowfolders deepfolders shallowfiles deepfiles
    (shallowfolders, deepfolders, shallowfiles, deepfiles)
end
let (shallowfolders, deepfolders, shallowfiles, deepfiles) = list_test()
    @test length(shallowfolders) ≥ 3
    @test length(deepfolders) ≥ 5
    @test length(shallowfiles) == 0
    @test length(deepfiles) == 2
    @info "==== (17) list_test ===="
end


function checksum_test()
    mp = Libaudio.modulepath(Libaudio)
    Libaudio.checksum_randinit(mp, "jl.sha")
    before = Libaudio.checksum_validate(mp, "jl.sha", ".jl", recursive=true)
    Libaudio.checksum_update(mp, "jl.sha", ".jl", recursive=true)
    after = Libaudio.checksum_validate(mp, "jl.sha", ".jl", recursive=true)
    (before, after)
end
let (before, after) = checksum_test()
    @test before == false
    @test after == true
    rm(joinpath(Libaudio.modulepath(Libaudio),"jl.sha"))
    @info "==== (18) checksum_test ===="
end




function sigdistratio_test()
    x = randn(8192)
    sdr = Libaudio.sigdistratio([zeros(1024); x; zeros(1024)], view(x,:))
    # @info "signal to distortion ratio (dB)" sdr
    return sdr
end
let sdr = sigdistratio_test()
    @test sdr > 160
    @info "==== (19) sigdistratio_test ===="
end


function resample_test()
    mp = Libaudio.modulepath(Libaudio)
    x, fs = Libaudio.wavread(joinpath(mp, "test/acqua_ieee_male_250ms_10450ms.wav"), Float64)
    y = Libaudio.resample_vhq(x, fs, 16000)
    # wavwrite(y, joinpath(mp, "test/acqua_ieee_male_250ms_10450ms_16khz.wav"), Fs=16000, nbits=32)
    z = Libaudio.resample_vhq(y, 16000, 48000)
    # wavwrite(z, joinpath(mp, "test/acqua_ieee_male_250ms_10450ms_48khz.wav"), Fs=48000, nbits=32)
    # set_zero_subnormals(true)
    sdr = Libaudio.sigdistratio([zeros(1024);view(z,:,1);zeros(1024)], view(x,:,1))
    # set_zero_subnormals(false)
    # @info "resample sdr value" sdr
    return sdr
end
let sdr = resample_test()
    @test sdr > 26.0
    @info "==== (20) resample_test ===="
end



function idealmask_test()
    path = Libaudio.modulepath(Libaudio)
    x1,fs = Libaudio.wavread(joinpath(path, "test/female_with_1s_silence_48khz.wav"), Float64)
    x2,fs = Libaudio.wavread(joinpath(path, "test/acqua_ieee_male_250ms_10450ms.wav"), Float64)
    binarymask, y1, y2 = Libaudio.idealsoftmask(x1[:,1], x2[:,1], fs)
    Libaudio.wavwrite(joinpath(path, "test/idealmask.wav"), [y1 y2], fs, 32)
    sdr1 = Libaudio.sigdistratio([zeros(8000);y1[:,1];zeros(8000)], x1[:,1])
    sdr2 = Libaudio.sigdistratio([zeros(8000);y2[:,1];zeros(8000)], x2[:,1])
    (sdr1, sdr2)
end
let (sdr1, sdr2) = idealmask_test()
    # @info "ideal mask test" sdr1 sdr2
    @test sdr1 > 13
    @test sdr2 > 11
    @info "==== (21) idealmask_test ===="
end


