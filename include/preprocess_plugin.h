#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>

class PreprocessPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    PreprocessPlugin(int srcW, int srcH, int dstW, int dstH);
    PreprocessPlugin(const void* data, size_t length);

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex,
        const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos,
        const nvinfer1::PluginTensorDesc* inOut,
        int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
        int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out,
        int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
        int nbInputs, const nvinfer1::PluginTensorDesc* outputs,
        int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override;
    nvinfer1::DataType getOutputDataType(int index,
        const nvinfer1::DataType* inputTypes,
        int nbInputs) const noexcept override;
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* ns) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    int srcW_, srcH_, dstW_, dstH_;
    std::string namespace_;
};

class PreprocessPluginCreator : public nvinfer1::IPluginCreator {
public:
    PreprocessPluginCreator();
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char* name,
        const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name,
        const void* data, size_t length) noexcept override;
    void setPluginNamespace(const char* ns) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection fc_;
    std::string namespace_;
};
