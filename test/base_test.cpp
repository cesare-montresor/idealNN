#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "Layers/Sequential.h"
#include "Layers/Dense.h"


TEST_CASE("Base Test") {
    Sequential *model = new Sequential();

    model->add( new Dense() );
    model->add( new Dense() );
    model->add( new Dense() );
    model->add( new Dense() );
    model->add( new Dense() );

    model->forward();

    REQUIRE(true);
}
