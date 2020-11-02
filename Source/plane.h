#include <iostream>
#include <glm/glm.hpp>

using glm::vec4;

class Plane4D {
  public:

    Plane4D(){}

    Plane4D( vec4 point, vec4 normal ) {
      this->point = point;
      this->normal = normal;
    }

    vec4 getPoint() { return point; }

    vec4 getNormal() { return normal; }

    bool isInPlane( vec4 v ) { return glm::dot( ( v - point ), normal ) >= 0; }

    vec4 intersectionPoint( vec4 v1, vec4 v2 ) {
      if ( ! ( isInPlane(v1) ^ isInPlane(v2) ) ) {
        printf("Illegal arguments in isInPlane(...)\n");
        exit(1);
      }

      return planeIntersection( v1, v2 );
    }

    // Returns 0 if both are out, 1 if both are in, 2 if only v1 is in, 3 if only v2 is in
    int intersects( vec4 v1, vec4 v2 ) {
      if ( !isInPlane( v1 ) && !isInPlane(v2) ) return 0;
      if (  isInPlane( v1 ) &&  isInPlane(v2) ) return 1;
      if (  isInPlane( v1 ) && !isInPlane(v2) ) return 2;
      return 3;
    }

  private:
    vec4 point;
    vec4 normal;

    vec4 planeIntersection( vec4 v1, vec4 v2 ) {

        float d1 = glm::dot( v1 - point, normal );
        float d2 = glm::dot( v2 - point, normal );
        float t = d1 / ( ( d1 - d2 ) );

        vec4 result;
        result = v1 + t * ( v2 - v1 );
        return result;
    }

};
