# flow-chain

A three-stage pipeline: `producer` hands data to `transform`, and
`transform` publishes through `publisher`. The stages import each
other in order; `producer` does not import `publisher` directly.
