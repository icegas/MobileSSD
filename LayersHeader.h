#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/normalize_layer.hpp"

namespace caffe
{
	extern INSTANTIATE_CLASS(NormalizeLayer);
}