### Scoring functions

function WIS(fc::forecast, h::Int64, returnSingle::Bool = false)
    w0 = 1/2
    α = sort(2 .* fc.quantile[fc.quantile .< 0.5])
    K = length(α)
    w = α ./ 2
    hind = findfirst(fc.horizon .== h)

    out = zeros(4)
    out[1] = w0 * abs(fc.truth[hind] - fc.Qmat[findfirst(fc.quantile .== 0.5), hind])
    for k = 1:K
        ind1 = argmin(abs.(fc.quantile .- α[k]/2))
        ind2 = argmin(abs.(fc.quantile .- (1 - α[k]/2)))
        l = fc.Qmat[ind1, hind]
        u = fc.Qmat[ind2, hind]
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
    hind = findfirst(fc.horizon .== h)
    out = 0.0
    Fq = [0; fc.quantile; 1]
    xq = [-Inf; fc.Qmat[:, hind]; Inf]
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