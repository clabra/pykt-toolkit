# Interpretable Transformers

## @di2025ante

- Prototype-based methods are based on optimizing a distance in a latent space ... e.g. using a distance-based classification algorithm in a low-dimensional learned feature space.
- Concept-based methods involve optimizing the weights of a layer through the use of regularization terms added to the loss function

## @mohebbi2024transformer

- interpretation of attention and context mixing:

  • Attention analysis (Clark et al., 2019) as
  a straightforward starting point for mea-
  suring context mixing.

  • Limitations of interpreting raw attention
  scores (Bibal et al., 2022; Hassid et al., 2022)

  • Effective attention scores: rollout (Ab-
  nar and Zuidema, 2020), HTA (Brun-
  ner et al., 2020), LRP-based attention
  (Chefer et al., 2020).

  • Expanding the scope of context mix-
  ing analysis by incorporating other
  model components: Attention-Norm
  (Kobayashi et al., 2020, 2021, 2023),
  GlobEnc (Modarressi et al., 2022), ALTI
  (Ferrando et al., 2022b,a), Value Zero-
  ing (Mohebbi et al., 2023b), DecompX
  (Modarressi et al., 2023).

## @li2020latent

Let an arbitrary AE be represented by z = F (x) and
x′ = G(z), where F (·) is the encoder, G(·) is the decoder, z
is the latent vector encoding x, and x′ is the reconstruction
of x (see Fig. 3). Note that when x′ is a good approx-
imation of x (i.e., x′ ≈ x), the attribute information of
x represented in y will also be captured in the latent en-
coding z. Attribute manipulation means that we replace
the attributes y captured by z with new attributes yn. Let
K(·) be a replacement function, then we have new latent
space zn = K(z, yn) and xn = G(K(z, yn)), where the
attribute information encoded in yn can be predicted from
xn and the non-attribute information of x will be preserved.
To give a concrete example, given an image of a face, x,
we wish to manipulate x w.r.t. the presence or absence of a
set of desired attributes encoded in yn (e.g., a face with or
without smiles, wearing or not wearing glasses), producing
the manipulated image xn, without changing the identify of
the face (i.e., preserving the non-attribute information of x)
...
We propose a matrix projection plugin that can be attached
to various autoencoders (e.g. Seq2seq, VAE) to make the
latent space factorised and disentangled, based on labelled
attribute information, which ensures that manipulation in
the latent space is much easier.
