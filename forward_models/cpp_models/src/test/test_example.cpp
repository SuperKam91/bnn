/* external codebase */
#include "Catch2/catch.hpp"
/* in-house code */
#include "example.hpp"

TEST_CASE("example_function","[example][function]") 
{
    REQUIRE( example_function(3) == 10 );
    REQUIRE( example_function(2) == 8 );
    REQUIRE( example_function(-1) == 2 );
    REQUIRE( example_function(0) == 4 );
}
