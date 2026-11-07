Upstream code: https://github.com/wjq-learning/CBraMod
Two core model files copied from the official GitHub repository on 2026-11-08:
- models/cbramod.py (GitHub blob SHA 8a52c243651c5e7df31e6e9017a57811f74db496)
- models/criss_cross_transformer.py (GitHub blob SHA 8f76374c54d0e60bebd75ba42c0de389bd56d91d)
The criss_cross_transformer.py copy contains small non-functional trims to unused helper documentation.
Official repository and MIT license belong to the original authors; model architecture is not our contribution.
Our new decoder, preprocessing and evaluation scripts live in src/ and scripts/.
Pretrained weights are NOT included in this ZIP. Download them with scripts/fetch_weights.py.
