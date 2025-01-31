### Scoring functions

function WIS(fc::forecast, h::Int64, returnSingle::Bool = false)
    hind = findall(fc.horizon .== h)
    if length(hind) == 0
        error("No forecasts found for horizon")
    else
        hind = hind[1]
    end
    w0 = 1/2
    α = getQuantiles(fc)[fc.horizon .== h][1]
    Qmat = getQmat(fc)
    K = length(α)
    w = α ./ 2
    
    out = zeros(4)
    out[1] = w0 * abs(fc.truth[hind] - fc.median[hind])
    
    for k = 1:K
        ind1 = argmin(abs.(quant .- α[k]/2))
        ind2 = size(Qmat)[1] - ind1 + 1
        l = Qmat[ind1, hind]
        u = Qmat[ind2, hind]
        y = fc.truth[hind]

        out[2] += w[k] * (u - l)
        out[3] += w[k] * 2/α[k]*(l - y)*(y < l)
        out[4] += w[k] * 2/α[k]*(y - u)*(y > u)
    end
    if returnSingle
        return out ./ (K + 1/2)
    else
        return sum(out) / (K + 1/2)
    end
end


function WIS(fc::forecast)
    (h -> WIS(fc, h)).(fc.horizon)
end


function QRPS(fc::forecast, h::Int64)
    hind = findall(fc.horizon .== h)
    if length(hind) == 0
        error("No forecasts found for horizon")
    else
        hind = hind[1]
    end
    out = 0.0
    α = getQuantiles(fc)[fc.horizon .== h][1]
    quants = [α ./ 2; reverse(1 .- α ./ 2)]
    Qmat = getQmat(fc)
    if length(fc.median) > 0
        quants = [quants; 0.5; reverse(1 .- quants)]
    else
        quants = [quants; reverse(1 .- quants)]
    end
    
    Fq = [0; quants; 1]
    xq = [-Inf; Qmat[:, hind]; Inf]
    y = fc.truth[hind]

    for i = 1:length(xq) - 1
        l = xq[i]
        u = xq[i+1]
        if y < l
            if isfinite(u)
                out += (1 - Fq[i]^2) * (u - l)
            end
        elseif y > u
            if isfinite(l)
                out += Fq[i]^2 * (u - l)
            end
        else
            if isfinite(l)
                out += Fq[i]^2 * (y - l) 
            end
            if isfinite(u)
               out += (1 - Fq[i])^2 * (u - y) 
            end
        end
    end
    out
end

function QRPS(fc::forecast)
    (h -> QRPS(fc, h)).(fc.horizon)
end