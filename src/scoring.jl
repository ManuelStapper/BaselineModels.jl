### Scoring functions

function WIS(fc::forecast, h::Int64;
             logTrafo::Bool = false,
             returnSingle::Bool = false)
    hind = findall(fc.horizon .== h)
    if length(hind) == 0
        error("No forecasts found for horizon")
    else
        hind = hind[1]
    end
    if length(fc.truth) == 0
        error("No truth data available")
    end
    w0 = 1/2
    α = getQuantiles(fc)[fc.horizon .== h][1]
    int = fc.interval[hind]

    K = length(α)
    w = α ./ 2

    truth = fc.truth
    med = fc.median
    if logTrafo
        truth = log.(truth .+ 1)
        int.l = log.(int.l .+ 1)
        int.u = log.(int.u .+ 1)
        med = log.(median .+ 1)
    end
    
    out = zeros(4)
    out[1] = w0 * abs(truth[hind] - med[hind])
    y = truth[hind]

    for k = 1:K
        l = int.l[k]
        u = int.u[k]
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


function WIS(fc::forecast; logTrafo::Bool = false)
    (h -> WIS(fc, h, logTrafo = logTrafo)).(fc.horizon)
end

function QRPS2(fc::forecast, h::Int64; logTrafo::Bool = false)
    hind = findall(fc.horizon .== h)
    if length(hind) == 0
        error("No forecasts found for horizon")
    else
        hind = hind[1]
    end
    if length(fc.truth) == 0
        error("No truth data available")
    end
    out = 0.0
    α = getQuantiles(fc)[hind]
    quants = [α ./ 2; reverse(1 .- α ./ 2)]
    int = fc.interval[hind]
    
    if length(fc.median) > 0
        quants = [quants; 0.5; reverse(1 .- quants)]
        xq = [-Inf; int.l; fc.median[hind]; reverse(int.u); Inf]
    else
        quants = [quants; reverse(1 .- quants)]
        xq = [-Inf; int.l; reverse(int.u); Inf]
    end

    truth = fc.truth
    if logTrafo
        truth = log.(truth .+ 1)
        xq = [-Inf; log.(xq[2:end-1] .+ 1); Inf]
    end
    
    Fq = [0; quants; 1]
    y = truth[hind]

    for i = 1:length(xq) - 1
        l = xq[i]
        u = xq[i+1]
        if y < l
            if isfinite(u)
                out += (1 - Fq[i])^2 * (u - l)
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

function QRPS(fc::forecast; logTrafo::Bool = false)
    (h -> QRPS(fc, h, logTrafo = logTrafo)).(fc.horizon)
end