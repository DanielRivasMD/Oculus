####################################################################################################

using Flux
using Flux: crossentropy

####################################################################################################

function trainCNN!(model, loader; hparams::CNNParams)
    loss(yh, y) = crossentropy(yh, y)
    opt = OptimiserChain(Descent(hparams.η), Momentum(hparams.momentum))
    st  = Flux.setup(opt, model)

    for epoch in 1:hparams.epochs
        for (xb, yb) in loader
            xb, yb = hparams.device(xb), hparams.device(yb)
            gs, = gradient(model) do m
                loss(m(xb), yb)
            end
            Flux.update!(st, model, gs)
        end
        @info "Finished epoch $epoch"
    end
end

####################################################################################################
