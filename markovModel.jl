using PyPlot
using Distributions

Δt = 0.01                    # ms
const T = 30.0
const t = collect(0.0:Δt:T)
pulseStart = 10.            # ms
pulseLen = 10.              # ms
PulseAmpl = 0.01             # nA

struct markovNeuron
    # state vector
    x::Array{Float64,1} # [v, p, i, n]

    # parameters
    g::Float64  # conductance            mS/cm^2
    E::Float64  # equilibrium potential  mV
    C::Float64  # Capacitance            μF/cm^2
    d::Float64  # diameter               μm
    A::Float64  # membrane area          cm^2
    N::Int64    # number of channels

end

"""
    Neuron Constructor
    Specify diameter in μm
    Other parameters inserted from Hodgkin & Huxley 1952
"""
function Neuron(d::Float64)

    gs = 36.0   # specific conductance mS/cm^2
    E = -12.0  # K+ equilibrium  mV
    Cs = 1.0    # specific capacitance μF/cm^2

    #
    A = π*d^2*1.0e-8  # membrane area in cm^2
    C = Cs*A          # neuron capacitance
    g = gs*A          # conductance
    v0 = 0.0          # initial voltage RMP
    p0 = 0.15         # initial open state prob
    i0 = 0.0          # initial current
    n0 = 20.0          # initial number of open channels
    x = [v0, p0, i0, n0]   # initial state
    N = round(Int64, g/10e-9)

  markovNeuron(x, g, E, C, d, A, N)

end

"""
  H-H rate parameters
"""
α(v) = v == 10.0 ?  0.1 : 0.01*(10.0-v)/(exp((10.0-v)/10.0)-1.0)
β(v) = 0.125*exp(-v/80.)


"""
    H-H state update
"""
function update(neuron, Inject, Δt)

    # copy the state variables
    v = neuron.x[1]
    p = neuron.x[2]
    n = neuron.x[4]
    N = neuron.N

    Inject = Inject*1.0e-3  # convert nA -> μA

    # coeffs of ODE for channel kinetics
    # τ dp/dt = p_inf - p
    τ = 1.0/(α(v) + β(v))
    p_infinity = α(v)*τ

    a = α(v)*Δt
    if a < 0.0
        a = 0.0
    elseif a > 1
        a = 1.0
    else
        a = a
    end

    b = β(v)*Δt
    if b < 0.0
        b = 0.0
    elseif b > 1
        b = 1.0
    else
        b = b
    end

    n_opening = round(rand(Binomial((N-n), a)))
    n_closing = round(rand(Binomial(n, b)))
    n_open = n + n_opening - n_closing

    Ichannel = n_open*10e-9*(v-neuron.E)

    # update state
    neuron.x[1] = v - Δt*(Ichannel-Inject)/neuron.C
    neuron.x[2] = p + Δt*(p_infinity - p)/τ
    neuron.x[3] = Ichannel*1.0e3    # convert μA -> nA
    neuron.x[4] = n_open

end


"""
  Pulse generator
"""
function pulse(t, start, len, amplitude)

    u = zeros(length(t))
    u[findall( t-> (t>=start) & (t<start+len), t)] .= amplitude

    return u
end

neuron = Neuron(1.)

# burn in
for i in 1:10000

    update(neuron, 0.0, Δt)

end

# input current
Inject = pulse(t, pulseStart, pulseLen, PulseAmpl)

# array to hold state
x = fill(0.0, length(t), length(neuron.x))
x[1,:] = neuron.x[:]

for i = 2:length(t)

    update(neuron, Inject[i], Δt)
    x[i, :] = neuron.x[:]

end

fig, (ax1, ax2, ax3, ax4, ax5) = subplots(nrows=5, ncols=1, figsize=(10,10))
ax1.plot(t, x[:,1])
ax1.set_title("Markov Model")
ax1.set_ylabel("mV re RMP")
ax1.set_xlim(0.0, T)
ax1.set_xlabel("Membrane Potential")

ax2.plot(t, x[:,2])
ax2.set_ylabel("Pr")
ax2.set_xlabel("Channel Open Probability")

ax3.plot(t, x[:,4])
ax3.set_ylabel("n")
ax3.set_xlabel("Number of Open Channels")

ax4.plot(t, x[:,3])
ax4.set_ylabel("nA")
ax4.set_xlabel("Channel Current")

ax5.plot(t, Inject)
ax5.set_ylabel("nA")
ax5.set_xlabel("Injected Current")




tight_layout()
display(fig)
savefig("HHmodel") # save as png
close(fig)
