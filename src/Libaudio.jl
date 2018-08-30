module Libaudio


using Polynomials
using LinearAlgebra
using Statistics
using FFTW
using SHA
using Random



modulepath(name) = realpath(joinpath(dirname(pathof(name)),".."))
"""
    init(module)

install binary dependencies to "C:\\Drivers\\Julia\\"
"""
function __init__()
    mkpath("C:\\Drivers\\Julia\\")
    cp(joinpath(modulepath(Libaudio), "deps/usr/lib/libsoxr.dll"), "C:\\Drivers\\Julia\\libsoxr.dll", force=true)
end




"""
    bilinear(b, a, fs)

bilinear transformation of transfer function from s-domain to z-domain via s = 2/T (z-1)/(z+1).

# Arguments
- 'b::AbstractVector': the numerator coefficient array, with decreasing order of s.
- 'a::AbstractVector': the denominator coefficient array, with decreasing order of s. 
- 'fs::Real': sample rate.

# Details
let 
```math
    ζ = z^{-1} we have s = \\frac{-2}{T} \\frac{ζ-1}{ζ+1} 

    H(s) = \\frac{b_m s^m + b_{m-1} s^{m-1} + ... + b_1 s + b_0}{a_n s^n + a_{n-1} s^{n-1} + ... + a_1 s + a_0}
```

So 
```math
    H(ζ) = \\frac{b_m (-2/T)^m (ζ-1)^m / (ζ+1)^m  + ... + b_1 (-2/T) (ζ-1)/(ζ+1) + b_0}{a_n (-2/T)^n (ζ-1)^n / (ζ+1)^n  + ... + a_1 (-2/T) (ζ-1)/(ζ+1) + a_0}
``` 

Since we assume H(s) is rational, so n ≥ m, multiply num/den with (ζ+1)^n ans we have
```math
    H(ζ) = \\frac{b_m (-2/T)^m (ζ-1)^m (ζ+1)^(n-m)  + b_{m-1} (-2/T)^{m-1} (ζ-1)^{m-1} (ζ+1)^{n-m+1} + ... + b_1 (-2/T) (ζ-1)(ζ+1)^{n-1} + b_0 (ζ+1)^n}
                 {a_n (-2/T)^n (ζ-1)^n  + a_{n-1} (-2/T)^{n-1} (ζ-1)^{n-1} (ζ+1) ... + a_1 (-2/T) (ζ-1)(ζ+1)^{n-1} + a_0 (ζ+1)^n}

    H(ζ) = \\frac{B[0] + B[1]ζ + B[2]ζ^2 + ... B[m]ζ^m}{A[0] + A[1]ζ + A[2]ζ^2 + ... A[n]ζ^n}
```

"""
function bilinear(b::AbstractVector, a::AbstractVector, fs::Real)::Tuple{Vector{Float64},Vector{Float64}}
    m = length(b) - 1
    n = length(a) - 1
    p = Poly{BigFloat}([0])
    q = Poly{BigFloat}([0])

    br = convert(AbstractVector{BigFloat}, reverse(b))
    ar = convert(AbstractVector{BigFloat}, reverse(a))

    for i = m:-1:0
        p = p + (br[i+1] * (BigFloat(-2fs)^i) * poly(ones(BigFloat,i)) * poly(-ones(BigFloat,n-i)))
    end
    for i = n:-1:0
        q = q + (ar[i+1] * (BigFloat(-2fs)^i) * poly(ones(BigFloat,i)) * poly(-ones(BigFloat,n-i)))        
    end
    
    num = zeros(n+1)
    den = zeros(n+1)
    for i = 0:n
        num[i+1] = p[i]   # array mutation involves implicit type conversion        
    end
    for i = 0:n
        den[i+1] = q[i]   
    end
    (num/den[1], den/den[1])
end




"""
    conv_naive(f, g)

discrete convolution in BigFloat precision.

# Arguments
- 'f::AbstractVector':
- 'g::AbstractVector':

# Details
commutativity:
```math
    (f*g)[n] = \\sum_{m=-\\inf}^{\\inf} f[m] g[n-m]
             = \\sum_{m=-\\inf}^{\\inf} f[n-m] g[m]
```
"""
function conv_naive(f::AbstractVector, g::AbstractVector)::Vector{BigFloat}
    m = length(f)
    n = length(g)
    l = m + n - 1
    y = zeros(BigFloat,l)
    
    for i = 0:l-1
        i1 = i
        dp = zero(BigFloat)    
        for j = 0:n-1
            ((i1>=0) & (i1<m)) && (dp += f[i1+1] * g[j+1])
            i1 -= 1
        end
        y[i+1] = dp
    end
    y
end



"""
    conv(u,v)

convolution of two vectors based on FFT algorithm.
element type of u and v belongs to LinearAlgebra.BlasFloat, which is
of type Union{Complex{Float32}, Complex{Float64}, Float32, Float64}.
element type of two vectors must be the same for the best performance.
"""
function conv(u::StridedVector{T}, v::StridedVector{T}) where T<:LinearAlgebra.BlasFloat
    nu = length(u)
    nv = length(v)
    n = nu + nv - 1
    np2 = n > 1024 ? nextprod([2,3,5], n) : nextpow(2, n)
    upad = [u; zeros(T, np2 - nu)]
    vpad = [v; zeros(T, np2 - nv)]
    if T <: Real
        p = plan_rfft(upad)
        y = irfft((p*upad).*(p*vpad), np2)
    else
        p = plan_fft!(upad)
        y = ifft!((p*upad).*(p*vpad))
    end
    return y[1:n]
end


"""
    xcorr(u,v)

compute the cross-correlation of two vectors.
"""
function xcorr(u, v)
    su = size(u,1); sv = size(v,1)
    if su < sv
        u = [u;zeros(eltype(u),sv-su)]
    elseif sv < su
        v = [v;zeros(eltype(v),su-sv)]
    end
    conv(u, reverse(conj(v), dims=1))
end


"""
    xcorrcoeff(s, x)

cross correlation with normalization, unlike xcorr which will zero-pad both argument
to the same length, this function results in strictly length(s) + length(x) - 1
"""
function xcorrcoeff(s::AbstractVector, x::AbstractVector)
    ns = length(s)
    nx = length(x)
    n = max(nx, ns)
    nsm1 = ns-1
    xe = [zeros(eltype(x), nsm1); x; zeros(eltype(x), nsm1)]
    y = zeros(promote_type(eltype(x), eltype(s)), nsm1+nx)

    kernel = sum(s.^2)
    k = 1
    for i = nsm1+nx:-1:1
        p = view(xe,i:i+nsm1)
        @inbounds y[k] = dot(p, s) / (sqrt(kernel * sum(p.^2)) + eps(eltype(s)))
        k += 1
    end
    y
end


function xcorrcoeff_threaded(s::AbstractVector{T}, x::AbstractVector{T}) where T<:LinearAlgebra.BlasFloat
    ns = length(s)
    nx = length(x)
    nsx = ns + nx
    n = max(nx, ns)
    nsm1 = ns-1
    xe = [zeros(T, nsm1); x; zeros(T, nsm1)]
    y = zeros(T, nsm1+nx)

    kernel = sum(s.^2)
    epsilon = eps(eltype(s))

    Threads.@threads for i = nsm1+nx:-1:1
        # p = view(xe,i:i+nsm1)
        dp = zero(T)
        @fastmath @simd for k = 1:ns
            @inbounds dp += xe[i-1+k] * s[k]
        end
        sp = zero(T)
        @fastmath @simd for k = i:i+nsm1
            @inbounds sp += xe[k]^2
        end
        # y[nsx-i] = dot(p,s) / (sqrt(kernel * sum(p.^2)) + epsilon)
        y[nsx-i] = dp/(sqrt(sp*kernel)+epsilon)
    end
    y
end



"""
    weighting_a(fs)

create a-weighting filter in z-domain that can be used by filter() function
"""
function weighting_a(fs)::Tuple{Vector{Float64},Vector{Float64}}
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    p = [ ((2π*f4)^2) * (10^(A1000/20)), 0, 0, 0, 0 ]
    q = conv_naive(Vector{BigFloat}([1, 4π*f4, (2π*f4)^2]), Vector{BigFloat}([1, 4π*f1, (2π*f1)^2]))
    q = conv_naive(conv_naive(q, Vector{BigFloat}([1, 2π*f3])), Vector{BigFloat}([1, 2π*f2]))
    num_z, den_z = bilinear(p, q, fs)
end







"""
    filt(b, a, x)

transfer function filter in z-domain

# Details

  y(n)        b(1) + b(2)Z^(-1) + ... + b(M+1)Z^(-M)
--------- = ------------------------------------------
  x(n)        a(1) + a(2)Z^(-1) + ... + a(N+1)Z^(-N)

  y(n)a(1) = x(n)b(1) + b(2)x(n-1) + ... + b(M+1)x(n-M)
             - a(2)y(n-1) - a(3)y(n-2) - ... - a(N+1)y(n-N)

"""
function filt(b::AbstractVector, a::AbstractVector, x::AbstractVecOrMat)
    if a[1] != 1.0
        b = b / a[1]
        a = a / a[1]
    end

    nb = length(b)
    na = length(a)
    m = nb-1
    n = na-1
    br = reverse(b)
    as = a[2:end]
    nx2 = size(x,2)

    y = zeros(size(x))
    x = [zeros(m, nx2); x]
    s = zeros(n, nx2)
    nx1 = size(x,1)

    if n != 0 #ARMA
        Threads.@threads for j = 1:nx2
            @inbounds for i = m+1:nx1
                y[i-m,j] = dot(br, view(x,i-m:i,j)) - dot(as, view(s,:,j))
                # ---------------------------
                # to move elements of array note the direction!
                @simd for k = n:-1:2
                    s[k,j] = s[k-1,j]
                end
                # s[2:end,j] = s[1:end-1,j]
                # ---------------------------
                s[1,j] = y[i-m,j]
            end
        end
    else #MA
        Threads.@threads for j = 1:nx2
            for i = m+1:nx1
                @inbounds y[i-m,j] = dot(br, view(x, i-m:i, j))
                # for k = 1:nb
                #     y[i-m,j] += br[k] * x[i-m-1+k,j]
                # end
                # [observation]: dot() is a better implement than loop
            end
        end
    end
    return y
end




"""
    hamming(n, flag="symmetric")

use "periodic" for STFT anaysis and synthesis
"""
function hamming(n::Integer, flag="symmetric")::Vector{Float64}
    lowercase(flag) == "periodic" && (n += 1)
    ω = [0.54 - 0.46cospi(2(i-1)/(n-1)) for i = 1:n]
    lowercase(flag) == "periodic" && (return ω[1:end-1])
    ω
end


"""
    hann(n, flag="symmetric")

use "periodic" for STFT anaysis and synthesis
"""
function hann(n::Integer, flag="symmetric")::Vector{Float64}
    lowercase(flag) == "periodic" && (n += 1)
    ω = [0.5 - 0.5cospi(2(i-1)/(n-1)) for i = 1:n]
    lowercase(flag) == "periodic" && (return ω[1:end-1])
    ω
end


"""
    sqrthann(n)

squared root hann widnow with asymmetric layout
"""
sqrthann(n) = sqrt.(hann(n,"periodic"))



"""
    WindowFrame

WindowFrame(fs, block, update)
WindowFrame(block, update)
"""
struct WindowFrame
    rate::Float64
    block::Int
    update::Int
    overlap::Int
    WindowFrame(r, b, u, o) = b < u ? error("block size must ≥ update size!") : new(r, b, u, b-u)
end
WindowFrame(fs, block, update) = WindowFrame(fs, block, update, 0)
WindowFrame(block, update) = WindowFrame(-1.0, block, update, 0)





"""
    zeropend(x, p, zeroprepend=false, zeroappend=false)

create extended version of array x with prepending/appending zeros for framing as well
as count the number of frames available. like 'convert()' if all zero-pendings are false
there would be no allocation of x.

# Arguments
- 'x::AbstractVector{T}': array of numbers for extension
- 'p::WindowFrame': instance of window frame parameter struct
- 'zeroprepend=false': bool switch for zero prepending
- 'zeroappend=false': bool swith for zero appending 

# Details
this is an utility function used by getframes(),spectrogram()...
new data are allocated, so original x is untouched.
zeroprepend = true: the first frame will have zeros of length nfft-nhop
zeroappend = true: the last frame will partially contain data of original x.
"""
function zeropend(x::AbstractVector{T}, p::WindowFrame, zeroprepend=false, zeroappend=false) where T<:Number
    zeroprepend && (x = [zeros(T, p.overlap); x])                                  
    length(x) < p.block && error("signal length must be at least one block!")       
    n = div(length(x) - p.block, p.update) + 1
    
    if zeroappend
        m = rem(length(x) - p.block, p.update)
        if m != 0
            x = [x; zeros(T, p.update-m)]
            n += 1
        end
    end
    (x,n)
    # n -> total number of frames to be processed
end




"""
    buffer(x, p, zeroprepend=false, zeroappend=false, window=ones)

Buffer a signal into a data frame. defualt window function is boxcar

# example:
    x = collect(1.0:100.0)
    p = WindowFrame(17, 7)
    y,h = buffer(x, p) where h is the unfold length in time domain 
"""
function buffer(x::AbstractVector{T}, p::WindowFrame, zeroprepend=false, zeroappend=false, window=ones) where T<:Number
    xp, n = zeropend(x, p, zeroprepend, zeroappend)
    ω = convert(AbstractVector{T}, window(p.block))  # default "symmetric"
    y = zeros(T, p.block, n)
    for i = 0:n-1
        s = p.update * i
        y[:,i+1] = ω .* view(xp, s+1:s+p.block)
    end
    # (p.update * n) is the total hopping size, +(p.block-p.update) for total length
    (y, p.update*n+(p.block-p.update))
end





"""
    spectrogram(x, p, zeroprepend=false, zeroappend=false, window=ones, nfft=p.block)

# Example:
    y,h = Libaudio.spectrogram(collect(1.0:100.0), Libaudio.WindowFrame(16,12), true, true) 
    where h is the unfold length in time domain
    note that Octave equivalent: fft(buffer(1:100, 16, 4))
"""
function spectrogram(x::AbstractVector{T}, p::WindowFrame, zeroprepend=false, zeroappend=false, window=ones, nfft=p.block) where T<:Number
    nfft < p.block && error("nfft length must be greater than or equal to block/frame length")
    x, n = zeropend(x, p, zeroprepend, zeroappend)
    m = div(nfft,2)+1
    ω = convert(AbstractVector{T}, window(nfft))
    t = plan_rfft(ω)
    𝕏 = zeros(Complex{T}, m, n)
    if nfft == p.block
        for i = 0:n-1
            s = p.update * i
            𝕏[:,i+1] = t * (ω .* view(x, s+1:s+p.block))
        end
    else
        for i = 0:n-1
            s = p.update * i
            𝕏[:,i+1] = t * (ω .* [view(x, s+1:s+p.block); zeros(T,nfft-p.block)])
        end
    end
    (𝕏, p.update*n+(p.block-p.update))
end



energy(v) = v.^2
intensity(v) = abs.(v)

function shortterm(f, x::AbstractVector{T}, p::WindowFrame, zeroprepend=false, zeroappend=false) where T<:Number
    frames, lu = buffer(x, p, zeroprepend, zeroappend)
    n = size(frames,2)
    ste = zeros(T, n)
    for i = 1:n
        ste[i] = sum(f(view(frames,:,i))) 
    end
    ste
end



binarysign(x) = x > 0 ? 1.0 : -1.0
zerocrossingrate(v) = floor.((abs.(diff(binarysign.(v)))) ./ 2)
ppnorm(v) = (v - minimum(v)) ./ (maximum(v) - minimum(v))
stand(v) = (v - mean(v)) ./ std(v)
hz2mel(hz) = 2595 * log10(1 + hz * 1.0 / 700)
mel2hz(mel) = 700 * (10 ^ (mel * 1.0 / 2595) - 1)
sigmoid(x::T) where T <: Number = one(T) / (one(T) + exp(-x))
sigmoidinv(x::T) where T <: Number = log(x / (one(T)-x))  # x ∈ (0, 1)
rms(x,dim) = sqrt.(sum((x.-mean(x,dim)).^2,dim)/size(x,dim))
rms(x) = sqrt(sum((x-mean(x)).^2)/length(x))



"""

calculate power spectrum of 1-D array on a frame basis
note that T=Float16 may not be well supported by FFTW backend

"""
function powerspectrum(x::AbstractVector{T}, p::WindowFrame, zeroprepend=false, zeroappend=false, window=ones, nfft=p.block) where T<:LinearAlgebra.BlasFloat
    nfft < p.block && error("nfft length must be greater than or equal to block/frame length")
    xp, n = zeropend(x, p, zeroprepend, zeroappend)
    ω = convert(AbstractVector{T}, window(nfft))
    f = plan_rfft(ω)
    m = div(nfft,2)+1
    ℙ = zeros(T, m, n)
    ρ = T(1 / nfft)
    if nfft == p.block
        for i = 0:n-1
            s = p.update * i
            ξ = f * (ω .* view(xp, s+1:s+p.block)) # typeof(ξ) == Array{Complex{T},1} 
            ℙ[:,i+1] = ρ * abs2.(ξ)
        end
    else
        for i = 0:n-1
            s = p.update * i
            ξ = f * (ω .* [view(xp, s+1:s+p.block); zeros(T,nfft-p.block)])
            ℙ[:,i+1] = ρ * abs2.(ξ)
        end
    end
    (ℙ, p.update*n + (p.block-p.update))
end




function localmaxima(x::AbstractVector)
    gtl = [false; x[2:end] .> x[1:end-1]]
    gtu = [x[1:end-1] .>= x[2:end]; false]
    imax = gtl .& gtu
    # imax -> Boolean BitArray
end



"""
    parabolicfit2(y)

# Details
    % given three points (x, y1) (x+1, y2) and (x+2, y3) there exists 
    % the only parabolic-2 fit y = ax^2+bx+c with a < 0. 
    % Therefore a global miximum can be found over the fit.
    %
    % To fit all thress points: let (x1 = 0, y1 = 0) then we have
    % other points (1, y2-y1) and (2, y3 - y1).
    %       0 = a * 0^2 + b * 0 + c => c = 0    (1)
    %       y2 - y1 = a + b                     (2)
    %       y3 - y1 = 4 * a + 2 * b             (3)
    %
    %       => a = y3/2 - y2 + y1/2             (4)
    %       => b = -y3/2 + 2*y2 - 3/2*y1        (5)

# Example
    % validation
    % y = [0.5; 1.0; 0.8];
    % [a,b] = parabolic_fit_2(y);
    % figure; plot([0;1;2], y-y(1), '+'); hold on; grid on;
    % m = -0.5*b/a;
    % ym = a * m^2 + b * m;
    % plot(m, ym, 's');
    % x = -1:0.001:5;
    % plot(x,a*x.^2+b*x, '--');
"""
function parabolicfit2(y::AbstractVector)
    a = 0.5y[3] - y[2] + 0.5y[1]
    b = -0.5y[3] + 2y[2] - 1.5y[1]
    (a,b)
end



"""
    extractsymbol(x::AbstractVector{T}, s::AbstractVector{T}, rep::Integer, dither=-160; vision=true, verbose=false)

extract symbols based on correlation coefficients.

# Arguments
- 'x::AbstractVector{T}': signal that contains the symbol
- 's::AbstractVector{T}': symbol template
- 'rep::Integer': number of symbols can be found within the signal
- 'dither': dB value for dithering, default to -160dB
"""
function extractsymbol(x::AbstractVector{T}, s::AbstractVector{T}, rep::Integer, dither=-160; vision=true, verbose=true, xcorrcoeff=false) where T<:Real
    x = x + (rand(T,size(x)) .- T(0.5)) * T(10^(dither/20))
    n = length(x) 
    m = length(s)
    y = zeros(T, rep * m)
    peaks = zeros(Int64, rep)
    lbs = zeros(Int64, rep)
    peakspf2 = zeros(rep)

    # do the cross correlation and sort the maxima
    if xcorrcoeff
        ℝ = xcorrcoeff_threaded(s, x)
    else
        ℝ = xcorr(s, x)
    end
    verbose && (@info "peak value" maximum(ℝ))
    # vision && (box = plot(x))
    𝓡 = sort(ℝ[localmaxima(ℝ)], rev=true)
    isempty(𝓡) && (return (y, diff(peaks)))

    # find the anchor point
    ploc = findfirst(isequal(𝓡[1]), ℝ)
    peaks[1] = ploc
    lb = n - ploc + 1
    rb = min(lb + m - 1, length(x))
    y[1:1+rb-lb] = x[lb:rb]
    ip = 1
    lbs[ip] = lb
    1+rb-lb < m && (@warn "incomplete segment extracted!")

    pf2a, pf2b = parabolicfit2(ℝ[ploc-1:ploc+1])
    peakspf2[ip] = (ploc-1) + (-0.5pf2b/pf2a)
    verbose && (@info "peak anchor-[$(ip)] in correlation" ploc peakspf2[ip])

    # if vision
    #     box_hi = maximum(x[lb:rb])
    #     box_lo = minimum(x[lb:rb])
        
    #     plot!(box,[lb,rb],[box_hi, box_hi], color = "red", lw=1)
    #     plot!(box,[lb,rb],[box_lo, box_lo], color = "red", lw=1)
    #     plot!(box,[lb,lb],[box_hi, box_lo], color = "red", lw=1)
    #     plot!(box,[rb,rb],[box_hi, box_lo], color = "red", lw=1)
    # end

    if rep > 1
        for i = 2:length(𝓡)
            ploc = findfirst(isequal(𝓡[i]), ℝ)
            if sum(abs.(peaks[1:ip] .- ploc) .> m) == ip
                ip += 1
                peaks[ip] = ploc

                pf2a, pf2b = parabolicfit2(ℝ[ploc-1:ploc+1])
                peakspf2[ip] = (ploc-1) + (-0.5pf2b/pf2a)            
                verbose && (@info "peak anchor-[$ip] in correlation" ploc, peakspf2[ip])

                lb = n - ploc + 1
                rb = min(lb + m - 1, length(x))
                lbs[ip] = lb
                
                # if vision
                #     box_hi = maximum(x[lb:rb])
                #     box_lo = minimum(x[lb:rb])    
                #     plot!(box,[lb,rb],[box_hi, box_hi], color = "red", lw=1)
                #     plot!(box,[lb,rb],[box_lo, box_lo], color = "red", lw=1)
                #     plot!(box,[lb,lb],[box_hi, box_lo], color = "red", lw=1)
                #     plot!(box,[rb,rb],[box_hi, box_lo], color = "red", lw=1)
                # end

                y[1+(ip-1)*m : 1+(ip-1)*m+(rb-lb)] = x[lb:rb]
                1+rb-lb < m && (@warn "incomplete segment extracted!")
                
                if ip == rep
                    break
                end
            end
        end
        peaks = sort(peaks)
        lbs = sort(lbs)
        peakspf2 = sort(peakspf2)
    end
    # vision && display(box)
    return (lbs, peaks, peakspf2, y)
end




"""
    dB20uPa(calibration, measurement, symbol, repeat, p, symbollow=0.0, symbolhigh=0.0, fl=100, fh=12000, calibratorreading=114.0; verbose=true)

# Arguments
    - 'calibration::AbstractVector{T}': calibration recording vector
    - 'measurement::AbstractMatrix{T}': measured recording vector 
    - 'symbol::AbstractVector{T}': symbol template vector
    - 'repeat::Integer': number of symbols to be found in measurement
    - 'p::WindowFrame': window frame settings
    - 'symbollow = 0.0': symbol stating time
    - 'symbolhigh = 0.0': symbol stop time
    - 'fl = 100': lower frequency bound
    - 'fh = 12000': higher frequency bound
    - 'calibratorreading = 114.0': calibrator reading for reference spl
    - 'verbose = true': default reveal everything
"""
function db20upa(
    calibration::AbstractVector{T}, 
    measurement::AbstractMatrix{T}, 
    symbol::AbstractVector{T}, 
    repeat::Integer,
    p::WindowFrame, 
    symbollow = 0.0, 
    symbolhigh = 0.0, 
    fl = 100, 
    fh = 12000, 
    calibratorreading = 114.0; 
    verbose = true) where T<:Real


    # calculate dbspl of all channels of x
    x = measurement
    s = symbol
    channels = size(x,2)
    dbspl = zeros(eltype(x), channels)
    
    rp,rpn = powerspectrum(calibration, p, false, false, hann)
    rp = mean(rp, dims=2)
    hl = floor(Int, fl/p.rate * p.block)
    hh = floor(Int, fh/p.rate * p.block)
    offset = 10log10(sum(view(rp,hl:hh)) + eps(eltype(rp)))

    # to use whole symbol, set symbollow >= symbolhigh
    if symbollow < symbolhigh
        @assert size(s,1) >= floor(Int, p.rate * symbolhigh)
        s = s[1+floor(Int, symbollow*p.rate) : floor(Int, symbolhigh*p.rate)]        
    end

    for c = 1:channels
        lbs, pk, pkpf, xp = extractsymbol(x[:,c], s, repeat)
        verbose && (@info "lb location in time and sample:" lbs./p.rate lbs)
        xps,xpn = powerspectrum(xp, p, false, false, hann)
        xpsu = mean(xps, dims=2)
                
        dbspl[c] = 10log10(sum(view(xpsu,hl:hh))) + (calibratorreading-offset)
        verbose && (@info "channel $c spl in dB" dbspl[c])           
    end
    return dbspl
end





"""
    spl(calibration, fs, measurement, symbol, repeat, p, symbollow=0.0, symbolhigh=0.0, fl=100, fh=12000, calibratorreading=114.0; weighting="none")

wrapper for db20upa: locate the calibration wav file and do proper weighting of the signals

# Arguments
    - 'calibration::AbstractVector{T}': calibration recording vector
    - 'measurement::AbstractMatrix{T}': measured recording vector 
    - 'symbol::AbstractVector{T}': symbol template vector
    - 'repeat::Integer': number of symbols to be found in measurement
    - 'p::WindowFrame': use WindowFrame(fs, 16384, div(16384,4)) for ACQUA compatible
    - 'symbollow = 0.0': symbol stating time
    - 'symbolhigh = 0.0': symbol stop time
    - 'fl = 100': lower frequency bound
    - 'fh = 12000': higher frequency bound
    - 'calibratorreading = 114.0': calibrator reading for reference spl
    - 'weighting="none"': "A" or "a" for a-wighting

# Details
    to measure single simple symbol: symbol_start = 0.0, symbol_stop = 0.0 then measure multiple simple symbols: 
    concatenate them into one symbol, and use symbol_start and symbol_stop as labelings for each iteration

# Example
    calibration, fs = wavread("/path/to/calibration.wav")
    Libaudio.spl(view(calibration,:,1), measurement, symbol, 3, Libaudio.WindowFrame(fs,16384,16384÷4), 0.0, 0.0, 100, 12000, 114.0, weighting="A")
"""
function spl(
    calibration::AbstractVector,
    measurement::AbstractMatrix, 
    symbol::AbstractVector, 
    repeat::Integer,
    p::WindowFrame,
    symbollow=0.0,
    symbolhigh=0.0,
    fl = 100, 
    fh = 12000, 
    calibratorreading = 114.0;
    weighting = "none")

    if lowercase(weighting) == "a"
        b,a = weighting_a(p.rate)
        r = filt(b, a, calibration)
        x = filt(b, a, measurement)
        s = filt(b, a, symbol)
        @info "spl: apply a-wighting"
    else
        r = calibration
        x = measurement
        s = symbol
    end
    db20upa(r, x, s, repeat, p, symbollow, symbolhigh, fl, fh, calibratorreading)    
end









function expsinesweep(hzl, hzh, t, fs)
    f0 = convert(BigFloat, hzl)
    f1 = convert(BigFloat, hzh)
    n = round(Int, t * fs)
    m = (f1 / f0) ^ (1 / n)
    Δ = 2pi * f0 / fs
    y = zeros(BigFloat, n)

    #calculate the phase increment gain
    #closed form --- [i.play[pauseSps] .. i.play[pauseSps + chirpSps - 1]]
    ϕ = zero(BigFloat)
    for k = 1:n
        y[k] = ϕ
        ϕ += Δ
        Δ = Δ * m
    end
    
    #the exp sine sweeping time could be non-integer revolutions of 2 * pi for phase phi.
    #Thus we find the remaining and cut them evenly from each sweeping samples as a constant bias.
    Δ = -mod2pi(y[n])
    Δ = Δ / (n - 1)
    ϕ = zero(BigFloat)

    for k = 1:n
        y[k] = sin(mod2pi(y[k] + ϕ))
        ϕ += Δ
    end
    return y
end



"""
    expsinesweep_i(ess, hzl, hzh)

gain compensated inverse exponential sine sweep signal, i.e. conv(ess, ess_i) gives a dirac impulse function

# Arguments
- 'ess::AbstractVector': exponential sine sweep signal
- 'f0': start frequency in hz
- 'f1': stop frequency in hz
"""
function expsinesweep_i(ess::AbstractVector, hzl, hzh)
    f0 = convert(BigFloat, hzl)
    f1 = convert(BigFloat, hzh)
    n = length(ess)
    atten = 20log10(BigFloat(0.5)) * log2(f1/f0) / (n-1)
    gain = zero(BigFloat)
    y = reverse(ess)
    for i = 1:n
        y[i] *= 10^(gain/20+1)
        gain += atten
    end
    return y
end






"""
    impulse(ess, decay, f0, f1, fs, recording)

decode impulse response (transfer function) based on exponential sine sweep and its responses.

# Arguments
- 'ess::AbstractVector': exponential sine sweep signal (without decay)
- 'decay::Integer': number of decaying sample slots for capture of trailing dynamics
- 'f0': ess start frequency in hz
- 'f1': ess stop frequency in hz
- 'fs': sample frequency 
- 'recording::AbstractMatrix': mic recordings as capture of the responses

# Example
    note: an example showing clock-drift effect
        x = LibAudio.sinesweep_exp(22, 22000, 10, 47999.5)
        y = LibAudio.sinesweep_exp(22, 22000, 10, 48000.0)
        r = zeros(length(y),1)
        r[:,1] = y
        f1,h1,d1,t1 = LibAudio.impresp(x, length(y)-length(x), 22, 22000, 48000, r)
        f2,h2,d2,t2 = LibAudio.impresp(x, 0, 22, 22000, 48000, r)
        plot(f1)
        plot!(f2)

        f1 is the "skewed" dirac, which is caused by clock drift.
"""
function impulse(ess::AbstractVector, decay::Integer, hzl, hzh, fs, recording::AbstractMatrix)
    ess_i = expsinesweep_i(ess, hzl, hzh)         # iess -> BigFloat precision
    u = convert(AbstractVector{Float64}, ess_i)
    x = convert(AbstractVector{Float64}, ess)
    y = convert(AbstractMatrix{Float64}, recording)
    
    m = length(x)
    p = m + decay
    @assert size(y,1) == p
    nfft = nextpow(2, m+p-1)

    v = fftn(u, nfft)
    dirac = real(ifftn(fftn(x, nfft) .* v, nfft))/nfft
    measure = real(ifftn(fftn(y, nfft) .* v, nfft))/nfft

    offset = (m/fs) / log(hzh/hzl)
    d12 = round(Int, log(2) * offset * fs)
    #fundamental = zeros(nfft-(m-div(d12,2)-1), size(y,2))
    #harmonic = zeros(m-div(d12,2), size(y,2))
    fundamental = measure[m-div(d12,2):end, :]
    harmonic = measure[1:m-div(d12,2), :]
    return (fundamental, harmonic, dirac, measure)
end




"""
fft and ifft on columns with zero-padding to length n
if x is vector ndims(x) will be raised to 2
if x is matrix then dimension remains
"""
fftn(x::Union{AbstractVector, AbstractMatrix},n::Integer) = fft([x; zeros(eltype(x), n-size(x,1), size(x,2))], 1)
ifftn(x::Union{AbstractVector, AbstractMatrix},n::Integer) = ifft([x; zeros(eltype(x), n-size(x,1), size(x,2))], 1)




"""
    symbol_expsinesweep(f0, f1, t, fs)

generate symbol for signal synchronization

# Details
    % sync symbol is the guard symbol for asynchronous recording/playback
    % 'asynchronous' means playback and recording may happen at different 
    % devices: for example, to measure mix distortion we play stimulus from
    % fireface and do mic recording at the DUT (with only file IO in most 
    % cases).
    %
    % we apply one guard symbol at both beginning and the end of the
    % session. visualized as:
    %
    % +-------+--------------------+-------+
    % | guard | actual test signal | guard |
    % +-------+--------------------+-------+
    %
    % NOTE:
    % The guard symbol shall contain not only the chirp but also a
    % sufficiently long pre and post silence. post silence is for the chirp
    % energy to decay, not to disturb the measurement; pre silence is to
    % prepare enough time for DUT audio framework context switching 
    % (products may have buggy glitches when change from system sound to
    % music). Our chirp signal is designed to have zero start and zero end
    % so it is safe to (pre/a)ppend zeros (no discontinuity issue).
    % 
    % 
    % typical paramters could be:
    %   f0 = 1000
    %   f1 = 1250
    %   elapse = 2.5
    %   fs = 48000
"""
function symbol_expsinesweep(f0, f1, t, fs)
    x1 = convert(AbstractVector{Float64}, expsinesweep(f0, f1, t, fs))
    x2 = -reverse(x1)
    [x1; x2[2:end]] # guarantee the contiunity
end






# signal is either ::Vector{Float64} or ::Matrix{Float64}
# the result will be dimensionally raised up to matrix
"""
    encode_syncsymbol(t_context, symbol, t_decay, signal, fs, channel=1, symbolgain=-6)

encode signal with syncsymbol.

# Details
    % this function encode the content of the stimulus for playback if sync
    % (guard) symbols are needed for asynchronous operations.
    %
    % +----------+------+-------+--------------------+------+-------+
    % | t_contxt | sync | decay |       signal       | sync | decay |
    % +----------+------+-------+--------------------+------+-------+
    % t_switch is time for DUT context switch
    % decay is inserted to separate dynamics of sync and the test signal
    %
    % for example:
    %     signal = [randn(8192,1); zeros(65536,1)];
    %     g = sync_symbol(1000, 1250, 1, 48000) * (10^(-3/20));
    %     y = add_sync_symbol(signal, 3, g, 2, 48000);
    %
    % we now have a stimulus of pre-silence of 3 seconds, guard chirp of
    % length 1 second, chirp decaying marging of 2 seconds, a measurement
    % of random noise.
"""
function encode_syncsymbol(t_context, symbol::AbstractVector, t_decay, signal::Union{AbstractVector, AbstractMatrix}, fs, channel=1, symbolgain=-6)
    s = 10^(symbolgain/20) * symbol
    n = size(signal,2)
    
    switch = zeros(round(Int, t_context * fs), n)
    sym = zeros(length(s), n)
    sym[:, channel] = s
    decay = zeros(round(Int, t_decay * fs), n)
    [switch; sym; decay; signal; sym; decay]
end



"""
return the vector of index indicating the start of the signal in each channel of 'encoded'

# Arguments
- 'encoded::AbstractMatrix': encoded signal
- 'symbol::AbstractVector': symbol template
- 't_decay': symbol decay time in seconds
- 't_signal': signal time in seconds, for ess case it involves the active length and the ess decaying
- 'fs': sample rate
"""
function decode_syncsymbol(encoded::AbstractMatrix, symbol::AbstractVector, t_decay, t_signal, fs)
    n = size(encoded,2)
    locat = zeros(Int,2,n)
    for i = 1:n
        lbs, pks, pks2, mgd = extractsymbol(view(encoded,:,i), symbol, 2)
        locat[:,i] = lbs
    end

    delta_measure = view(locat,2,:) - view(locat,1,:)
    delta_theory = length(symbol) + round(Int, t_decay * fs) + round(Int, t_signal * fs)
    delta_measure_relative = view(locat,1,:) .- minimum(view(locat,1,:))
    @info "decode_syncsymbol delta" delta_measure delta_theory delta_measure_relative

    #lb = lbs[1] + size(symbol,1) + round(Int, t_decay * fs)
    #rb = lbs[2] - 1
    locat[1,:] .+ length(symbol) .+ round(Int, t_decay * fs)
end






"""
calculate filter banks in Mel domain

# Example
    using Plots
    mf = Libaudio.melfilterbanks(16000, 1024)
    plot(mf')
"""
function melfilterbanks(rate, nfft::Integer, filt::Integer=26, fl=0.0, fh=rate/2)
    fh > rate/2 && error("high frequency must be less than or equal to nyquist frequency!")
    ml = hz2mel(fl)
    mh = hz2mel(fh)
    mel_points = range(ml, length=filt+2, stop=mh)
    hz_points = mel2hz.(mel_points)

    # round frequencies to nearest fft bins
    𝕓 = floor.(Int, (hz_points/rate) * (nfft+1))
    #print(𝕓)

    # first filterbank will start at the first point, reach its peak at the second point
    # then return to zero at the 3rd point. The second filterbank will start at the 2nd
    # point, reach its max at the 3rd, then be zero at the 4th etc.
    𝔽 = zeros(filt, div(nfft,2)+1)
    for i = 1:filt
        for j = 𝕓[i]:𝕓[i+1]
            𝔽[i,j+1] = (j - 𝕓[i]) / (𝕓[i+1] - 𝕓[i])
        end
        for j = 𝕓[i+1]:𝕓[i+2]
            𝔽[i,j+1] = (𝕓[i+2] - j) / (𝕓[i+2] - 𝕓[i+1])
        end
    end
    𝔽m = 𝔽[vec(.!(isnan.(sum(𝔽,dims=2)))),:]
    return 𝔽m
end




function filterbankenergy_mel(x::AbstractVector{T}, p::WindowFrame, zeroprepend=false, zeroappend=false, window=ones, filt=26, fl=0, fh=div(p.rate,2); use_log=true) where T<:Real
    ℙ,h = powerspectrum(x, p, zeroprepend, zeroappend, window)
    𝔽 = melfilterbanks(p.rate, p.block, filt, fl, fh)
    ℙ = 𝔽 * ℙ
    use_log && (log.(ℙ.+eps(T)))
    ℙ
end






"""
    stft2(x, p, window)

analysis window [example: square-root hann] based on short-time fourier transform

# Arguments
- 'x::AbstractVector{T}': input time series
- 'p::WindowFrame': block/hop size
- 'window': window function to use

# Note
zero prepending and appending are applied to the input vector
"""
function stft2(x::AbstractVector{T}, p::WindowFrame, window=sqrthann) where T<:Real
    𝕏,h = spectrogram(x, p, true, true, window)
    𝕏,h
    # 𝕏 -> complex STFT output (DC to Nyquist)
    # h -> unpacked sample length of the signal in time domain
end



"""
    stft2(𝕏, h, p, window)

synthesis window [example: square-root hann] based on short-time fourier transform

# Arguments
- '𝕏::AbstractMatrix{Complex{T}}': complex spectrogram (DC to Nyquist)
- 'h   unpacked sample length of the signal in time domain
- 'p::WindowFrame': block/hop size
- 'window': window function to use

# Note
zero prepending and appending are applied to the input vector
"""
function stft2(𝕏::AbstractMatrix{Complex{T}}, h::Integer, p::WindowFrame, window=sqrthann) where T<:Real
    𝕎 = convert(AbstractVector{T}, window(p.block)) ./ convert(T, p.block/p.update)
    𝕏 = vcat(𝕏, conj!(𝕏[end-1:-1:2,:]))
    𝕏 = real(ifft(𝕏,1)) .* 𝕎
    y = zeros(T,h)
    for k = 0:size(𝕏,2)-1
        s = p.update * k
        y[s+1:s+p.block] .+= 𝕏[:,k+1]
    end
    y
end


"""
    idealsoftmask(x1, x2, fs)

demo function for soft-mask stft decomposition and sythesis

# Example
    path = Libaudio.modulepath(Libaudio)
    x1,fs = wavread(joinpath(path, "test/cleanspeech/sp02.wav"))
    x2,fs = wavread(joinpath(path, "test/cleanspeech/sp03.wav"))
    binarymask, y1, y2 = Libaudio.idealsoftmask(x1[:,1], x2[:,1], fs)
    sdr1 = Libaudio.sigdistratio([zeros(1000);y1[:,1];zeros(1000)], x1[:,1])
    sdr2 = Libaudio.sigdistratio([zeros(1000);y2[:,1];zeros(1000)], x2[:,1])
"""
function idealsoftmask(x1::AbstractVector, x2::AbstractVector, fs)
    x1 = view(x1,:,1)
    x2 = view(x2,:,1)
    m = min(length(x1), length(x2))
    x1 = view(x1,1:m)
    x2 = view(x2,1:m)
    x = x1 + x2

    p = WindowFrame(1024, div(1024,4))
    pmix, h0 = stft2(x, p, sqrthann)
    px1, h1 = stft2(x1, p, sqrthann)
    px2, h2 = stft2(x2, p, sqrthann)

    bm = abs.(px1) ./ (abs.(px1) + abs.(px2))
    py1 = bm .* pmix
    py2 = (1 .- bm) .* pmix

    scale = 2
    y = stft2(pmix, h0, p, sqrthann) * scale
    y1 = stft2(py1, h0, p, sqrthann) * scale
    y2 = stft2(py2, h0, p, sqrthann) * scale
    y = view(y,1:m)
    y1 = view(y1,1:m)
    y2 = view(y2,1:m)
    delta = 10log10(sum(abs.(x-y).^2)/sum(x.^2))
    bm,y1,y2
    #histogram(bm[100,:])
end
    















"""
# Examples
    - list(path) -> list all subfolders under 'path'
    - list(path, recursive=true)
    - list(path, ".wav") 
    - list(path, ".wav", recursive=true)
"""
function list(path::String, t=""; recursive=false)
    x = Vector{AbstractString}()
    if recursive
        for (root, dirs, files) in walkdir(path)
            for dir in dirs
                isempty(t) && push!(x, joinpath(root, dir))
            end
            for file in files
                !isempty(t) && occursin(lowercase(t), lowercase(file)) && push!(x, joinpath(root, file))
            end
        end
    else
        a = readdir(path)
        for i in a
            p = joinpath(path,i)
            isempty(t) && isdir(p) && push!(x, p)
            !isempty(t) && occursin(lowercase(t), i) && push!(x, p)
        end
    end
    x
end


"""
    checksum(list)

summation of sha256 of all files in the 'list' and return with array of uint8 of length 32
"""
function checksum(list::Vector{AbstractString})        
    d = zeros(UInt8, 32)
    for j in list
        d .+= open(j) do f
            sha256(f)
        end
    end
    d
end


"""
    checksum_randinit(path, label="wav.sha")

build initial checksum directly under 'path' with file name 'label'
successive update may cause checksum mismatch, which can be utilized to trigger more events
"""
checksum_randinit(path::AbstractString, label="wav.sha") = write(joinpath(path, label), sha256(randstring()))

    
"""
    checksum_update(path::String, label="wav.sha", t=".wav"; recursive=true)

update the 'label' with all files under 'path' with file extension 't'
"""    
checksum_update(path::AbstractString, label="wav.sha", t=".wav"; recursive=true) = write(joinpath(path, label), checksum(list(path, t, recursive=recursive)))



"""
    checksum_validate(path::AbstractString, label="wav.sha", t=".wav"; recursive=true)    

validate SHA256 of all files with type 't' under 'path' recursively or not, matches that given in 'label'
"""
function checksum_validate(path::AbstractString, label="wav.sha", t=".wav"; recursive=true)    
    p = read(joinpath(path, label))
    q = checksum(list(path, t, recursive=recursive))
    isequal(0,sum(p-q))
end



"""
    textgrid_prase(file::AbstractString)

parse TextGrid of Praat into event vector
"""
function textgrid_prase(file::AbstractString)
    result = Array{Tuple{String, Float64, Float64},1}()
    open(file, "r") do fid
        x = ""
        while !occursin(Regex("intervals: size = "), x)
            x = readline(fid)
        end
        while !eof(fid)
            x = readline(fid)  # interval [1]:
            xmin = readline(fid)
            xmax = readline(fid)
            text = readline(fid)
            if !isempty(text[21:end-2])
                push!(result, (text[21:end-2], parse(Float64, xmin[20:end]), parse(Float64, xmax[20:end])))
            end
        end
    end
    return result
end



"""
    resample_vhq(input, fi, fo)

resample 'input' vector from sample rate 'fi' to 'fo'

# Note
    This function relies on ccall to libsox.dll/.a, since ccall is not a function but a compiler
    command, the lib path must be string literal, which brings trouble to deployment. for now
    please don't forget to change the lib path for relocation.
"""
function resample_vhq(input::AbstractVector{T}, fi, fo) where T<:Real
    block = convert(Vector{Float32}, input)
    n = ceil(Int64, length(block) * (fo/fi))
    resampled = zeros(Float32, n)
    resampled_n_return = zeros(UInt64,1)

    soxerr = ccall((:soxr_oneshot, "C:\\Drivers\\Julia\\libsoxr"),
                    Ptr{Int8},
                    (Float64, Float64, UInt32, Ptr{Float32}, UInt64, Ptr{UInt64}, Ptr{Float32}, UInt64, Ptr{UInt64}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), 
                    Float64.(fi), Float64.(fo), 1, block, length(block), C_NULL, resampled, length(resampled), resampled_n_return, C_NULL, C_NULL, C_NULL)
    @assert Int(soxerr) == 0
    @info "libsox resample theory/actual samples" n Int(resampled_n_return[1])
    return convert(AbstractVector{T}, resampled)
end


"""
    wrapper for resample_vhq() that enables input as matrix, which is the case for imported wav files
"""
function resample_vhq(input::AbstractMatrix{T}, fi, fo) where T<:Real
    m,n = size(input)
    resampled = zeros(T, ceil(Int64, m * (fo/fi)), n)
    for i = 1:n
        resampled[:,i] = resample_vhq(view(input,:,i), fi, fo)
    end
    return resampled
end


"""
    sigdistratio(x, t)

calculate signal to distrotion ratio between the original signal template 't', and the to-be-measured signal 'x'
both must be in vector form. this function use correlation to align them before calculate the differences, so
please prepend/append zeros to x for robust operation. result will be in dB unit.

# Arguments
    - 'x::AbstractVector{T}': signal to be measured
    - 't::AbstractVector{T}': signal template, the golden reference
"""
function sigdistratio(x::AbstractVector{T}, t::AbstractVector{T}) where T<:AbstractFloat
    lbs, pks, pks2, y = extractsymbol(x, t, 1)
    sdr = 10log10.(sum(t.^2,dims=1)./sum((t-y).^2,dims=1))
    sdr[1]
end





end # module
