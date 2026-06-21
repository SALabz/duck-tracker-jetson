#include "preprocess_plugin.h"
#include "preprocess_kernel.h"
#include <cstring>
#include <stdexcept>

// ── Constructor ──────────────────────────────────────────────────────────────
PreprocessPlugin::PreprocessPlugin(int srcW, int srcH, int dstW, int dstH)
    : srcW_(srcW), srcH_(srcH), dstW_(dstW), dstH_(dstH) {}

PreprocessPlugin::PreprocessPlugin(const void* data, size_t length) {
    const int* d = reinterpret_cast<const int*>(data);
    srcW_ = d[0]; srcH_ = d[1]; dstW_ = d[2]; dstH_ = d[3];
}

// ── Clone ────────────────────────────────────────────────────────────────────
nvinfer1::IPluginV2DynamicExt* PreprocessPlugin::clone() const noexcept {
    return new PreprocessPlugin(srcW_, srcH_, dstW_, dstH_);
}

// ── Output shape: [1, 3, dstH, dstW] float32 ────────────────────────────────
nvinfer1::DimsExprs PreprocessPlugin::getOutputDimensions(int outputIndex,
    const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    nvinfer1::DimsExprs out;
    out.nbDims = 4;
    out.d[0] = exprBuilder.constant(1);
    out.d[1] = exprBuilder.constant(3);
    out.d[2] = exprBuilder.constant(dstH_);
    out.d[3] = exprBuilder.constant(dstW_);
    return out;
}

// ── Format: input uint8 NHWC, output float32 NCHW ───────────────────────────
bool PreprocessPlugin::supportsFormatCombination(int pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int nbInputs, int nbOutputs) noexcept {
    if (pos == 0) {
        return inOut[0].type == nvinfer1::DataType::kINT8 &&
               inOut[0].format == nvinfer1::TensorFormat::kLINEAR;
    }
    return inOut[1].type == nvinfer1::DataType::kFLOAT &&
           inOut[1].format == nvinfer1::TensorFormat::kLINEAR;
}

void PreprocessPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
    int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out,
    int nbOutputs) noexcept {}

size_t PreprocessPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
    int nbInputs, const nvinfer1::PluginTensorDesc* outputs,
    int nbOutputs) const noexcept { return 0; }

// ── enqueue: call your existing CUDA kernel ──────────────────────────────────
int PreprocessPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept {
    launchPreprocessKernel(
        (const uint8_t*)inputs[0],
        (float*)outputs[0],
        srcW_, srcH_, dstW_, dstH_,
        stream);
    return 0;
}

nvinfer1::DataType PreprocessPlugin::getOutputDataType(int index,
    const nvinfer1::DataType* inputTypes,
    int nbInputs) const noexcept { return nvinfer1::DataType::kFLOAT; }

const char* PreprocessPlugin::getPluginType() const noexcept { return "PreprocessPlugin"; }
const char* PreprocessPlugin::getPluginVersion() const noexcept { return "1"; }
int PreprocessPlugin::getNbOutputs() const noexcept { return 1; }
int PreprocessPlugin::initialize() noexcept { return 0; }
void PreprocessPlugin::terminate() noexcept {}

size_t PreprocessPlugin::getSerializationSize() const noexcept {
    return 4 * sizeof(int);
}

void PreprocessPlugin::serialize(void* buffer) const noexcept {
    int* d = reinterpret_cast<int*>(buffer);
    d[0] = srcW_; d[1] = srcH_; d[2] = dstW_; d[3] = dstH_;
}

void PreprocessPlugin::destroy() noexcept { delete this; }
void PreprocessPlugin::setPluginNamespace(const char* ns) noexcept { namespace_ = ns; }
const char* PreprocessPlugin::getPluginNamespace() const noexcept { return namespace_.c_str(); }

// ── Creator ──────────────────────────────────────────────────────────────────
PreprocessPluginCreator::PreprocessPluginCreator() {
    fc_.nbFields = 0;
    fc_.fields = nullptr;
}

const char* PreprocessPluginCreator::getPluginName() const noexcept { return "PreprocessPlugin"; }
const char* PreprocessPluginCreator::getPluginVersion() const noexcept { return "1"; }
const nvinfer1::PluginFieldCollection* PreprocessPluginCreator::getFieldNames() noexcept { return &fc_; }

nvinfer1::IPluginV2* PreprocessPluginCreator::createPlugin(const char* name,
    const nvinfer1::PluginFieldCollection* fc) noexcept {
    return new PreprocessPlugin(640, 480, 640, 640);
}

nvinfer1::IPluginV2* PreprocessPluginCreator::deserializePlugin(const char* name,
    const void* data, size_t length) noexcept {
    return new PreprocessPlugin(data, length);
}

void PreprocessPluginCreator::setPluginNamespace(const char* ns) noexcept { namespace_ = ns; }
const char* PreprocessPluginCreator::getPluginNamespace() const noexcept { return namespace_.c_str(); }
