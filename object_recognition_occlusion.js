// the animals
var animal_names = ["elric", "espere", "venti", "drue", "twan", "stylis"]

// features: furry, sharp teeth, wings
var feature_lookup = {
  "elric": {fur: 0.45, claw: 0.40, tail: 0.90, ears: 0.99, nose: 0.65, feathers: 0.01, spots: 0.10, horns: 0.30},
  "espere": {fur: 0.75, claw: 0.01, tail: 0.01, ears: 0.99, nose: 0.85, feathers: 0.40, spots: 0.10, horns: 0.05},
  "venti": {fur: 0.70, claw: 0.01, tail: 0.3, ears: 0.05, nose: 0.45, feathers: 0.99, spots: 0.10, horns: 0.01},
  "drue": {fur: 0.30, claw: 0.30, tail: 0.45, ears: 0.85, nose: 0.70, feathers: 0.01, spots: 0.99, horns: 0.50},
  "twan": {fur: 0.10, claw: 0.80, tail: 0.65, ears: 0.20, nose: 0.30, feathers: 0.05, spots: 0.60, horns: 0.10},
  "stylis": {fur: 0.90, claw: 0.35, tail: 0.20, ears: 0.99, nose: 0.85, feathers: 0.25, spots: 0.10, horns: 0.05},
}

// prob that (if feature there AND occlude) that we see it
var transp_lookup = {
  "elric": {fur: 0.70, claw: 0.20, tail: 0.35, ears: 0.50, nose: 0.80, feathers: 0.01, spots: 0.01, horns: 0.05},
  "espere": {fur: 0.90, claw: 0.01, tail: 0.01, ears: 0.65, nose: 0.40, feathers: 0.20, spots: 0.05, horns: 0.01},
  "venti": {fur: 0.85, claw: 0.01, tail: 0.2, ears: 0.01, nose: 0.10, feathers: 0.90, spots: 0.05, horns: 0.01},
  "drue": {fur: 0.50, claw: 0.10, tail: 0.10, ears: 0.25, nose: 0.10, feathers: 0.05, spots: 0.90, horns: 0.15},
  "twan": {fur: 0.05, claw: 0.55, tail: 0.65, ears: 0.05, nose: 0.10, feathers: 0.10, spots: 0.55, horns: 0.05},
  "stylis": {fur: 0.95, claw: 0.15, tail: 0.10, ears: 0.85, nose: 0.65, feathers: 0.20, spots: 0.05, horns: 0.01},
}

// the observations for the current stimulus 
// (can be "inaccurate" to account for occlusion)
var obs = {fur: true, claw: true, tail: true, ears: true, nose: true, feathers: true, spots: true, horns: true}

// define the generative model:
// 1. pick an animal at random (uniform prior, can be changed)
// 2. get the features of those animal from the feature lookup
// 3. condition on the features that you observe matching the observation
// 4. return the identity of the animal
var model = function() {
  
  // 1. 
  var animal = sample(Categorical({ps: [1,1,1,1,1,1], vs: animal_names}))
  
  // 2. 
  var has_fur =  feature_lookup[animal].fur * transp_lookup[animal].fur
  var has_claw = feature_lookup[animal].claw * transp_lookup[animal].claw
  var has_tail = feature_lookup[animal].tail * transp_lookup[animal].tail
  var has_ears = feature_lookup[animal].ears * transp_lookup[animal].ears
  
  var has_nose =  feature_lookup[animal].nose * transp_lookup[animal].nose
  var has_feathers = feature_lookup[animal].feathers * transp_lookup[animal].feathers
  var has_spots = feature_lookup[animal].spots * transp_lookup[animal].spots
  var has_horns = feature_lookup[animal].horns * transp_lookup[animal].horns
  
  // 3. 
  observe(Bernoulli({p: has_fur}), obs.fur)
  observe(Bernoulli({p: has_claw}), obs.claw)
  observe(Bernoulli({p: has_tail}), obs.tail)
  observe(Bernoulli({p: has_ears}), obs.ears)
  observe(Bernoulli({p: has_nose}), obs.nose)
  observe(Bernoulli({p: has_feathers}), obs.feathers)
  observe(Bernoulli({p: has_spots}), obs.spots)
  observe(Bernoulli({p: has_horns}), obs.horns)  
  // 4.
  animal
}

// use MCMC to infer the original animal, conditioned on the observation
var dist = Infer({method: "MCMC", samples: 100000, model: model})
viz(dist, {xLabel: 'Animal', yLabel: 'Probability'})
