#include "plane.h"
#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include "SDLauxiliary.h"
#include "TestModelH.h"
#include <stdint.h>
#include <algorithm>

using namespace std;
using glm::ivec2;
using glm::ivec3;
using glm::vec2;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

#define pi 3.1415926
#define SCREEN_WIDTH 320
#define SCREEN_HEIGHT 320
#define FULLSCREEN_MODE false
#define f 320

screen* scrn;

void initialisePlanes();
mat4 transformationMatrix();
mat4 getHomogenisationMatrix();
void interpolate( ivec2 a, ivec2 b, vector<ivec2>& result );

const float Z_OFFSET = ( (float)f * 2 ) / SCREEN_HEIGHT;
vec4 cameraPos( 0, 0, - 1 - Z_OFFSET, 1 );

float moveZ = 0;
float moveY = 0;
float moveX = 0;
float pitch = 0;
float yaw = 0;
float roll = 0;

float depthBuffer[SCREEN_HEIGHT][SCREEN_WIDTH];

Plane4D planes[6];
#define TOP 0
#define BOTTOM 1
#define RIGHT 2
#define LEFT 3
#define NEAR 4
#define FAR 5

vec4 originalLightPosition = vec4( 0, -0.5, -0.7, 1 );
vec4 lightPos = originalLightPosition;
vec3 lightPower = 6.0f * vec3( 1, 1, 1 ); // D
vec3 indirectLightPowerPerArea = 0.6f * vec3( 1, 1, 1 ); // N

vec4 currentNormal;
vec3 currentReflectance;

const mat4 homogenisationMatrix = getHomogenisationMatrix();

struct pixel {
  int x;
  int y;
  float zinv;
  vec4 pos3d;
};

struct vertex {
  vec4 position;
};

/* ----------------------------------------------------------------------------*/
/* FUNCTIONS                                                                   */

bool Update();
void Draw(screen* screen);

int main( int argc, char* argv[] ) {

  if( SDL_Init( SDL_INIT_VIDEO ) < 0) {
    fprintf( stderr, "Could not initialise SDL: %s\n", SDL_GetError() );
    exit( -1 );
  }

  initialisePlanes();
  screen *screen = InitializeSDL( SCREEN_WIDTH, SCREEN_HEIGHT, FULLSCREEN_MODE );

  while( Update() ) {

    Draw(screen);
    SDL_Renderframe(screen);
  }

  SDL_SaveImage( screen, "screenshot.bmp" );

  KillSDL(screen);
  return 0;
}

void initialisePlanes() {
  planes[TOP] = Plane4D( vec4( 0, 1, 0, 1 ), vec4( 0, -1, 0, 1 ) ); // Top
  planes[BOTTOM] = Plane4D( vec4( 0, -1, 0, 1 ), vec4( 0, 1, 0, 1 )); // Bottom

  planes[RIGHT] = Plane4D( vec4( 1, 0, 0, 1 ), vec4( -1, 0, 0, 1 ) ); // right
  planes[LEFT] = Plane4D( vec4( -1, 0, 0, 1 ), vec4( 1, 0, 0, 1 ) ); // left

  planes[NEAR] = Plane4D( vec4( 0, 0, 0.0001, 1 ), vec4( 0, 0, 1, 1 ) ); // near
  planes[FAR] = Plane4D( vec4( 0, 0, 4, 1 ), vec4( 0, 0, -1, 1 ) ); // far
}

void swap(int *x, int *y) {
  *x += *y;
  *y = *x - *y;
  *x = *x - *y;
}

// <interpolation functions>
void interpolate( ivec2 a, ivec2 b, vector<ivec2>& result ) {
  int x0 = a.x;
  int y0 = a.y;

  int x1 = b.x;
  int y1 = b.y;

  bool orderReverted = false;

  // Always work with x0 < x1
  if ( x0 > x1 ) {
    swap( &x0, &x1 );
    swap( &y0, &y1 );
    orderReverted = true;
  }

  int x = x0;
  int y = y0;

  int dx = x1 - x0;
  int dy = y1 - y0;

  // x changes faster than y
  bool xMain = abs( dx ) > abs( dy );
  // slope is positive
  bool mPos = y1 > y0;

  int steps = result.size();//xMain ? abs( dx ) : abs( dy );
  int change = mPos ? 1 : -1;

  int d = xMain ? (2 * change * dy - dx) : (2 * dx - change * dy);

  for ( int i = 0; i < steps; i++) {

    if ( orderReverted ) {
      result[steps - i - 1].x = x;
      result[steps - i - 1].y = y;
    } else {
      result[i].x = x;
      result[i].y = y;
    }

    if ( d < 0 ) {
      x += xMain;
      y += mPos ? !xMain : -!xMain;
      d += xMain ? 2 * change * dy : 2 * dx;
    } else {
      x += 1;
      y += change;
      d += xMain ? 2 * ( change * dy - dx ) : 2 * ( dx - change * dy );
    }

  }
}

void interpolate1Dfloat( float a, float b, vector<float>& result ) {
  int N = result.size();
  float step = ( b - a ) / (float)max( N - 1, 1 );
  float current = a;
  for (int i = 0; i < N; i++ ) {
    result[i] = current;
    current += step;
  }
}

void interpolate( vec4 a, vec4 b, vector<vec4>& result ) {
  a = a / a.z;
  b = b / b.z;

  int N = result.size();
  vec4 step = vec4( b - a ) / float( max( N - 1, 1 ) );
  vec4 current( a );
  for ( int i = 0; i < N; i++ ) {
    result[i] = current;
    result[i] = result[i] / result[i].w;
    current += step;
  }
}

void interpolate( pixel a, pixel b, vector<pixel>& result ) {
  size_t N = result.size();
  vector<ivec2> Pixels( N );
  vector<float> Depths( N );
  vector<vec4> _3dPositions( N );

  ivec2 a_(a.x, a.y );
  ivec2 b_( b.x, b.y );

  interpolate( a_, b_, Pixels );
  interpolate( a.pos3d, b.pos3d, _3dPositions );
  interpolate1Dfloat( a.zinv, b.zinv, Depths );

  for ( size_t i = 0; i < N; i++ ) {
    result[i].x = Pixels[i].x;
    result[i].y = Pixels[i].y;
    result[i].zinv = Depths[i];
    result[i].pos3d = _3dPositions[i];
  }
}
// </interpolation functions>

mat4 getHomogenisationMatrix() {
  mat4 matrix;

  matrix[0] = vec4( 1.0f,    0,           0, 0 );
  matrix[1] = vec4(    0, 1.0f,           0, 0 );
  matrix[2] = vec4(    0,    0,        1.0f, 0 );
  matrix[3] = vec4(    0,    0, 1.0f / 2.0f, 0 );
  matrix = glm::transpose( matrix );

  return matrix;
}

mat4 transformationMatrix() {
  float angleX = pitch * pi / 180;
  float angleY = yaw * pi / 180;
  float angleZ = roll * pi / 180;

  mat4 rotation(1.0f);
  mat4 rotationX(1.0f);
  mat4 rotationY(1.0f);
  mat4 rotationZ(1.0f);
  mat4 translation(1.0f);

  // ROTATION

  // Rotation about x axis
  rotationX[0] = vec4( 1, 0,              0,             0 );
  rotationX[1] = vec4( 0, cos( angleX ), -sin( angleX ), 0 );
  rotationX[2] = vec4( 0, sin( angleX ),  cos( angleX ), 0 );
  rotationX[3] = vec4( 0, 0,              0,             1 );
  rotationX = glm::transpose( rotationX );

  // Rotation about y axis
  rotationY[0] = vec4(  cos( angleY ), 0, sin( angleY ), 0 );
  rotationY[1] = vec4(  0,             1, 0,             0 );
  rotationY[2] = vec4( -sin( angleY ), 0, cos( angleY ), 0 );
  rotationY[3] = vec4(  0,             0, 0,             1 );
  rotationY = glm::transpose( rotationY );

  // Rotation about z axis
  rotationZ[0] = vec4( cos( angleZ ), -sin( angleZ ), 0, 0 );
  rotationZ[1] = vec4( sin( angleZ ),  cos( angleZ ), 0, 0 );
  rotationZ[2] = vec4( 0,              0,             1, 0 );
  rotationZ[3] = vec4( 0,              0,             0, 1 );
  rotationZ = glm::transpose( rotationZ );

  rotation = rotationX * rotationY * rotationZ;

  // TRANSLATION
  translation[0] = vec4( 1, 0, 0, -cameraPos.x + moveX);
  translation[1] = vec4( 0, 1, 0, -cameraPos.y + moveY);
  translation[2] = vec4( 0, 0, 1, -cameraPos.z + moveZ);
  translation[3] = vec4( 0, 0, 0, 1           );
  translation = glm::transpose( translation );

  return rotation * translation;

}

void vertexShader( const vertex& v, pixel& p ) {
  p.zinv = 1 / v.position.z;
  p.y = f * v.position.y * p.zinv + SCREEN_HEIGHT / 2;
  p.x = f * v.position.x * p.zinv + SCREEN_WIDTH / 2;
  p.pos3d = v.position;
}

int getPixels( pixel a, pixel b ) {

  ivec2 a_ = ivec2( a.x, a.y );
  ivec2 b_ = ivec2( b.x, b.y );

  ivec2 delta = glm::abs( a_ - b_ );
  return glm::max( delta.x, delta.y ) + 1;
}

int max3( int a, int b, int c ) {
  if ( max( a, b ) > c )
    return max( a, b );
  else
    return c;
}

int getRows( vector<pixel> vertices ) {
  vector<int> ys( 3 );
  vector<int> ds( 3 );

  for ( int i = 0; i < 3; i++ )
    ys[ i ] = (int)vertices[ i ].y;

  ds[0] = abs( ys[0] - ys[1] );
  ds[1] = abs( ys[0] - ys[2] );
  ds[2] = abs( ys[2] - ys[1] );

  int max = max3( ds[0], ds[1], ds[2] );

  return max + 1;
}

void computeTriangleOrientation( vector<pixel> vertexPixels, pixel& vLeft, pixel& vRight, pixel& v ) {
  int dy;

  for ( size_t i = 0; i < vertexPixels.size(); i++ ) {
    dy = vertexPixels[i].y - vertexPixels[( i + 1 ) % vertexPixels.size()].y;

    if ( dy == 0 ) {
      v = vertexPixels[( i + 2 ) % vertexPixels.size()];
      if ( vertexPixels[i].x < vertexPixels[( i + 1 ) % vertexPixels.size()].x ) {
        vLeft = vertexPixels[i];
        vRight = vertexPixels[( i + 1 ) % vertexPixels.size()];
      } else {
        vLeft = vertexPixels[( i + 1 ) % vertexPixels.size()];
        vRight = vertexPixels[i];
      }
      break;
    }
  }
}

void computePolygonRows(
      const vector<pixel> vertexPixels,
      vector<pixel>& leftPixels,
      vector<pixel>& rightPixels ) {

  vector<vector<pixel>> edges = *(new vector<vector<pixel>>( 2 ) );

  pixel vLeft, vRight, v; // v is the vertex not touching the flat edge

  computeTriangleOrientation( vertexPixels, vLeft, vRight, v );

  int p1 = getPixels( v, vLeft );
  int p2 = getPixels( v, vRight );

  edges[0].resize( p1 );
  edges[1].resize( p2 );

  // breshenham( v.x, v.y, vLeft.x, vLeft.y, edges[0] );
  // breshenham( v.x, v.y, vRight.x, vRight.y, edges[1] );
  interpolate( v, vLeft, edges[0] );
  interpolate( v, vRight, edges[1] );

  // Left
  uint32_t pixelIndex = 0;
  uint32_t prevIndex = 0;
  for ( int row = 0; row < (int)leftPixels.size(); row++ ) {

    leftPixels[row].x    = edges[0][pixelIndex].x;
    leftPixels[row].y    = edges[0][pixelIndex].y;
    leftPixels[row].zinv = edges[0][pixelIndex].zinv;
    leftPixels[row].pos3d = edges[0][pixelIndex].pos3d;

    prevIndex = pixelIndex;
    while ( pixelIndex < edges[0].size() && leftPixels[row].y == edges[0][pixelIndex].y ) pixelIndex++;

    if ( edges[0][pixelIndex - 1].x < edges[0][prevIndex].x ) {
          leftPixels[row].x    = edges[0][pixelIndex - 1].x;
          leftPixels[row].y    = edges[0][pixelIndex - 1].y;
          leftPixels[row].zinv = edges[0][pixelIndex - 1].zinv;
          leftPixels[row].pos3d = edges[0][pixelIndex - 1].pos3d;
    }
  }

  //Right
  pixelIndex = 0;
  for ( int row = 0; row < (int)rightPixels.size(); row++ ) {

    rightPixels[row].x   = edges[1][pixelIndex].x;
    rightPixels[row].y   = edges[1][pixelIndex].y;
    rightPixels[row].zinv = edges[1][pixelIndex].zinv;
    rightPixels[row].pos3d = edges[1][pixelIndex].pos3d;

    // pixelIndex++;
    while ( pixelIndex < edges[1].size() && rightPixels[row].y == edges[1][pixelIndex].y ) pixelIndex++;
  }
}

bool comareYs( pixel v1, pixel v2 ) { return v1.y < v2.y; }

void splitTriangle( vector<pixel> oldT, vector<pixel>& triangle1, vector<pixel>& triangle2 ) {
  std::sort( oldT.begin(), oldT.end(), comareYs );

  // if triangle is already flat
  if ( oldT[0].y == oldT[1].y || oldT[1].y == oldT[2].y ) {
    triangle1 = oldT;
    triangle2 = oldT;
    return;
  }

  int u4 = (int)( ( (float)( oldT[1].y - oldT[2].y ) ) / ( (float)( oldT[0].y - oldT[2].y ) ) * ( oldT[0].x - oldT[2].x ) ) + oldT[2].x;

  int pixels = getPixels( oldT[0], oldT[2] );
  vector<pixel> result( pixels );
  interpolate( oldT[0], oldT[2], result );
  size_t i = 0;
  while ( i < result.size() && ( result[i].x != u4 || result[i].y != oldT[1].y ) ) i++;

  if ( i == result.size() )  {
    i = 0;
    while ( i < result.size() && ( result[i].x != u4 && result[i].y != oldT[1].y ) ) i++;
  }

  triangle1[0] = oldT[0];
  triangle1[1] = oldT[1];
  triangle1[2].x = u4;
  triangle1[2].y = oldT[1].y;
  triangle1[2].zinv = result[i].zinv;
  triangle1[2].pos3d = result[i].pos3d;

  triangle2[0] = oldT[1];
  triangle2[1] = oldT[2];
  triangle2[2] = triangle1[2];

}

void drawPolygonRows ( vector<pixel> leftpx, vector<pixel> rightpx, vec3 color ) {
  vec3 toLightSource;
  vec3 D;
  vec3 illumination;

  for ( size_t i = 0; i < leftpx.size(); i++ ) {

    while ( i < leftpx.size()             &&
       !( rightpx[i].x < 2 * SCREEN_WIDTH &&
          rightpx[i].x >= 0               &&
          leftpx[i].x < SCREEN_WIDTH      &&
          leftpx[i].x >= - SCREEN_WIDTH   &&
          leftpx[i].x <= rightpx[i].x ) )
          i++;

     if ( i == leftpx.size() ) break;

    vector<pixel> row( rightpx[i].x - leftpx[i].x + 1 );
    interpolate( leftpx[i], rightpx[i], row );
    for ( size_t j = 0; j < row.size(); j++ ) {
      if (  leftpx[i].y < SCREEN_HEIGHT &&
            leftpx[i].y >= 0 &&
            leftpx[i].x + j < SCREEN_WIDTH &&
            leftpx[i].x + j >= 0 &&
            row[j].zinv > depthBuffer[leftpx[i].y][leftpx[i].x + j] ) {

        toLightSource = vec3(lightPos) - vec3( row[j].pos3d );
        float len = glm::length(toLightSource);
        D = lightPower * glm::max( glm::dot( toLightSource, vec3( currentNormal ) ), (float)0 ) / (float)( 4 * pi * glm::pow( len, 2 ) );
        illumination = currentReflectance * ( D + indirectLightPowerPerArea );

        PutPixelSDL( scrn, leftpx[i].x + j, leftpx[i].y, illumination );

        depthBuffer[leftpx[i].y][leftpx[i].x + j] = row[j].zinv;
      }
    }
  }
}

void drawPolygon( const vector<vertex> vertices, vec3 color ) {
  int V = vertices.size();
  int ROWS;

  vector<vector<pixel>> flatTriangles = *(new vector<vector<pixel>>( 2, vector<pixel>( 3 ) ) );
  vector<pixel> leftPixels;
  vector<pixel> rightPixels;
  vector<pixel> vertexPixels = *(new vector<pixel>( V ) );

  for( int i = 0; i < V; i++ )
    vertexShader( vertices[i], vertexPixels[i] );

  splitTriangle( vertexPixels, flatTriangles[0], flatTriangles[1] );

  for ( int i = 1; i > -1; i-- ) {
    ROWS = getRows( flatTriangles[i] );
    leftPixels.resize( ROWS );
    rightPixels.resize( ROWS );

    computePolygonRows( flatTriangles[i], leftPixels, rightPixels );
    drawPolygonRows( leftPixels, rightPixels, color );
  }
}

void unhomogenise( vector<vertex>& triangle ) {
  for ( uint8_t i = 0; i < triangle.size(); i++ )
    triangle[i].position.w = 1;
}

void homogenise( vector<vertex>& triangle ) {
  for ( uint8_t i = 0; i < triangle.size(); i++ )
    triangle[i].position = homogenisationMatrix * triangle[i].position;
}

// Assumes the quadrilateral lies on a 2D plane
// Assumes the vertices are ordered on a path on the circumference of the quad.
vector<vector<vertex>> splitQuadrilateralToTriangles( vector<vertex> quadrilateral ) {
  if ( quadrilateral.size() != 4 ) {
    printf("Unexpected argument size in splitQuadrilateralToTriangles(...)\n");
    exit(1);
  }
  vector<vector<vertex>> triangles;
  try {
    triangles = *( new vector<vector<vertex>>( 2, vector<vertex>( 3 ) ) );
  } catch (std::bad_alloc & ba)
  {
     std::cerr << "bad_alloc caught: " << ba.what();
     exit(1);
  }

  triangles[0][0] = quadrilateral[0];
  triangles[0][1] = quadrilateral[1];
  triangles[0][2] = quadrilateral[2];

  triangles[1][0] = quadrilateral[2];
  triangles[1][1] = quadrilateral[3];
  triangles[1][2] = quadrilateral[0];

  return triangles;
}

vector<vector<vertex>> splitToTriangles( vector<vertex> vertices ) {
  if ( !( vertices.size() == 3 || vertices.size() == 0 || vertices.size() == 4 ) ) {
    printf("Unexpected argument size in splitToTriangles(...): %lu\n", vertices.size());
    exit(1);
  }

  vector<vector<vertex>> triangles;
  try {
  triangles = *(new vector<vector<vertex>>());
  } catch (std::bad_alloc & ba)
  {
     std::cerr << "bad_alloc caught: " << ba.what();
     exit(1);
  }

  if ( vertices.size() == 3 ) {
    triangles.push_back(vertices);
    return triangles;
  }

  if ( vertices.size() == 0 ) {
    return triangles;
  }

  // vertices.size() == 4
  return splitQuadrilateralToTriangles( vertices );
}

vector<vector<vertex>> clipHomogenised( vector<vertex> originalT, int planeIndex ) {
  vertex v1, v2, intersectionPoint;
  // Stores the vertices of the polygon inside the frustum
  vector<vertex> verticesIn;
  vector<vector<vertex>> tempResult;
  vector<vector<vertex>> result;
  vector<vector<vertex>> resultBuffer;

  for ( size_t i = 0; i < originalT.size(); i++ ) {
    v1 = originalT[ i ];
    v2 = originalT[ ( i + 1 ) % originalT.size() ];

    switch ( planes[planeIndex].intersects( v1.position, v2.position ) ) {
      case 0: // both out
        break;
      case 1: // both in
        verticesIn.push_back( v2 );
        break;
      case 2: // v1 in
        intersectionPoint.position = planes[planeIndex].intersectionPoint( v1.position, v2.position );
        verticesIn.push_back( intersectionPoint );
        break;
      case 3: // v2 in
        intersectionPoint.position = planes[planeIndex].intersectionPoint( v1.position, v2.position );
        verticesIn.push_back( intersectionPoint );
        verticesIn.push_back( v2 );
        break;
      default:
        printf("Unexpected return value from intersectionWithTop(...)\n");
        exit(1);
    }
  }

  tempResult = splitToTriangles( verticesIn );

  if ( planeIndex < 5 ) {
    for ( unsigned char i = 0; i < tempResult.size(); i++ ) {
      resultBuffer = clipHomogenised( tempResult[i], planeIndex + 1 );
      for ( unsigned char j = 0; j < resultBuffer.size(); j++ ) {
        result.push_back( resultBuffer[j] );
      }
    }
  }
  else
    result = tempResult;

  return result;
}

vector<vector<vertex>> clip( vector<vertex> triangle ) {
  vector<vector<vertex>> triangles;

  homogenise( triangle );

  triangles = clipHomogenised( triangle, 0 );

  for ( size_t i = 0; i < triangles.size(); i++ )
    unhomogenise( triangles[i] );

  return triangles;
}

/*Place your drawing here*/
void Draw( screen* screen ) {
  memset( screen -> buffer, 0, screen -> height * screen -> width * sizeof( uint32_t ) );

  scrn = screen;
  lightPos = transformationMatrix() * originalLightPosition;

  // Vertices of all the potential triangles after clipping
  vector<vector<vertex>> vertices;
  vector<vertex> originalT( 3 );
  vector<Triangle> triangles;
  LoadTestModel(triangles);

  memset( depthBuffer, 0, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof( float ) );

  for ( size_t i = 0; i < triangles.size(); i++ ) {

    currentNormal = triangles[i].normal;
    currentReflectance = triangles[i].color;

    for ( size_t j = 0; j < 3; j++ )
      originalT[j].position = transformationMatrix() * triangles[i].v(j);

    vertices = clip( originalT );

    for ( size_t j = 0; j < vertices.size(); j++ )
      drawPolygon( vertices[j], triangles[i].color );
  }
}

bool Update() {
  int dx, dy;
  SDL_GetRelativeMouseState( &dx, &dy );

  yaw   -= dx;
  pitch += dy;

  SDL_Event e;
  while(SDL_PollEvent(&e)) {

      if (e.type == SDL_QUIT) {
	       return false;
	    } else
      	if (e.type == SDL_KEYDOWN) {
      	    int key_code = e.key.keysym.sym;
      	    switch(key_code) {
              case SDLK_RIGHT:
                roll += 1;
                break;
              case SDLK_LEFT:
                roll -= 1;
                break;
              case SDLK_DOWN:
                moveY -= 0.1;
                break;
              case SDLK_UP:
                moveY += 0.1;
                break;
              case SDLK_w:
                moveZ -= 0.1;
                break;
      	      case SDLK_s:
                moveZ += 0.1;
                break;
      	      case SDLK_a:
                moveX += 0.1;
      		      break;
      	      case SDLK_d:
                moveX -= 0.1;
                break;
      	      case SDLK_ESCAPE:
      		      return false;
            }
        }
  }
  return true;
}
