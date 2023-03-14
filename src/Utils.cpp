//
// Created by cesare on 12/02/23.
//

#include <Common.h>
#include <Tensor/Tensor.h>
#include <Layer/LinearLayer.h>
#include <Utils.h>

namespace IdealNN::Utils{

    TensorArrayRef slice(const TensorArrayRef& array, ArrayIndex start, ArraySize count){
        auto as = getSize(array);

        if(start + 1 >= as ) {return Tensor::MakeTensorArray();}
        auto items_count = start + count < as ? count : (as - start);

        auto first = array->begin() + start;
        auto last = array->begin() + start + items_count;

        return Tensor::MakeTensorArray(TensorArray{first, last});
    }


    bool ScalarValueEqual(ScalarValue a, ScalarValue b){ return a - b < ScalarDelta; }

    ScalarValueArrayRef MakeScalarValueArray() { return std::make_shared<ScalarValueArray>(); }
    ScalarValueArrayRef MakeScalarValueArray(ArraySize size) { return std::make_shared<ScalarValueArray>(size); }
    ScalarValueArrayRef MakeScalarValueArray(ScalarValueArray scalarValueArray) { return std::make_shared<ScalarValueArray>(std::move(scalarValueArray)); }

    TensorDataRef MakeTensorData(ArraySize rows, ArraySize cols) { return std::make_shared<TensorData>(rows, cols); }
    TensorDataRef MakeTensorData(TensorData &&tensorData) { return std::make_shared<TensorData>(std::move(tensorData)); }

    TensorDataArrayRef MakeTensorDataArray() { return std::make_shared<TensorDataArray>(); }
    TensorDataArrayRef MakeTensorDataArray(ArraySize size) { return std::make_shared<TensorDataArray>(size); }
    TensorDataArrayRef MakeTensorDataArray(TensorDataArray tensorDataArray) { return std::make_shared<TensorDataArray>(std::move(tensorDataArray)); }

    LayerArrayRef MakeLayerArray(){ return std::make_shared<LayerArray>(); }
    LayerArrayRef MakeLayerArray(ArraySize size){ return std::make_shared<LayerArray>(size); }
    LayerArrayRef MakeLayerArray(LayerArray layerArray){ return std::make_shared<LayerArray>(std::move(layerArray)); }


    //TODO: make a log function that can be globally enabled/disabled
    //void log(){
        //https://stackoverflow.com/questions/3649278/how-can-i-get-the-class-name-from-a-c-object
    //}
}


