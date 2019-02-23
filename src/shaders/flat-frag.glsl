#version 300 es
precision highp float;

const vec3 u_Eye = vec3(0., 0., -10.);
const vec3 u_Ref = vec3(0.);
const vec3 u_Up = vec3(0., 1., 0.);

uniform vec2 u_Dimensions;
uniform float u_Time;

in vec2 fs_Pos;
out vec4 out_Col;

// SHADER GLOBAL VARIABLES
vec3 final_Pos;
int color_Id = -1;
const vec3 light_Vec = vec3(4.0, -2.0, -2.0);
const vec3 light_Vec_Col = vec3(4., 16., 43.) / 255.;
const vec3 light_Vec2 = vec3(-4.0, 2., -2.);
const vec3 light_Vec2_Col = vec3(117., 100., 127.) / 255.;
const vec3 point_Light = vec3(-12., 9.5, 14.3);
const vec3 point_Light_Base = vec3(255., 200., 25.) / 255.;
vec3 point_Light_Col;

// Define color/shader IDs.
#define BACKGROUND 0
#define WALLS 1
#define WOOD 2
#define BED 3
#define BEDCOVER 4
#define LAMPSHADE 5
#define LAMPPOST 6
#define DOOR 7
#define EYES 8

bool floatEquality(float x, float y) {
	return abs(x - y) < 0.0001;
}

float dot2( in vec3 v ) {
	return dot(v,v);
}

/*
 * GENERAL SDFs AND SDF OPERATIONS
 */

float sphereSDF(vec3 p, float r) {
	return length(p) - r;
}

float boxSDF(vec3 p, vec3 b) {
  vec3 d = abs(p) - b;
  return length(max(d,0.0))
         + min(max(d.x,max(d.y,d.z)),0.0); // remove this line for an only partially signed sdf 
}

float cappedConeSDF(vec3 p, float h, float r1, float r2) {
	vec2 q = vec2(length(p.xz), p.y);
    vec2 k1 = vec2(r2,h);
    vec2 k2 = vec2(r2 - r1, 2.0 * h);
    vec2 ca = vec2(q.x - min(q.x,(q.y < 0.0)?r1:r2), abs(q.y)-h);
    vec2 cb = q - k1 + k2*clamp( dot(k1-q,k2)/dot(k2, k2), 0.0, 1.0 );
    float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s * sqrt( min(dot(ca, ca),dot(cb, cb)) );
}

float cappedCylinderSDF(vec3 p, vec2 h)
{
  vec2 d = abs(vec2(length(p.xz), p.y)) - h;
  return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float opUnion(float d1, float d2) {
	return min(d1, d2);
}

float opSmoothUnion( float d1, float d2, float k ) {
    float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h); }


float opSubtraction(float d1, float d2) {
	return max(-d1, d2);
}

float opIntersection(float d1, float d2) {
	return max(d1, d2);
}

vec3 translate(vec3 p, float x, float y, float z) {
	return vec3( mat4(vec4(1.0, 0, 0, 0),
					  vec4(0, 1.0, 0, 0),
					  vec4(0, 0, 1.0, 0),
					  vec4(x, y, z, 1.0))
				 * vec4(p, 1.0) );
}

vec3 rotateX(vec3 p, float degrees) {
	return vec3( mat4(vec4(1.0, 0, 0, 0),
					  vec4(0, cos(radians(degrees)), -sin(radians(degrees)), 0),
					  vec4(0, sin(radians(degrees)), cos(radians(degrees)), 0),
					  vec4(0, 0, 0, 1.0))
				 * vec4(p, 1.0) );
}

vec3 rotateY(vec3 p, float degrees) {
	return vec3( mat4(vec4(cos(radians(degrees)), 0, sin(radians(degrees)), 0),
					   vec4(0, 1.0, 0, 0),
					   vec4(-sin(radians(degrees)), 0, cos(radians(degrees)), 0),
					   vec4(0, 0, 0, 1.0))
				 * vec4(p, 1.0) );
}

vec3 rotateZ(vec3 p, float degrees) {
	return vec3( mat4(vec4(cos(radians(degrees)), -sin(radians(degrees)), 0, 0),
					  vec4(sin(radians(degrees)), cos(radians(degrees)), 0, 0),
					  vec4(0, 0, 1.0, 0),
					  vec4(0, 0, 0, 1.0))
				 * vec4(p, 1.0) );
}

/*
 * NOISE / TOOLBOX FUNCTIONS
 */

float noise(float i) {
	return fract(sin(vec2(203.311f * float(i), float(i) * sin(0.324f + 140.0f * float(i))))).x;
}

float random(vec2 p, vec2 seed) {
  return fract(sin(dot(p + seed, vec2(127.1, 311.7))) * 43758.5453);
}

float interpNoise1D(float x) {
	float intX = floor(x);	
	float fractX = fract(x);

	float v1 = noise(intX);
	float v2 = noise(intX + 1.0f);
	return mix(v1, v2, fractX);
}

float fbm(float x) {
	float total = 0.0f;
	float persistence = 0.5f;
	int octaves = 8;

	for(int i = 0; i < octaves; i++) {
		float freq = pow(2.0f, float(i));
		float amp = pow(persistence, float(i));

		total += interpNoise1D(x * freq) * amp;
	}

	return total;
}

float interpNoise2D(float x, float y) {
	float intX = floor(x);
	float fractX = fract(x);
	float intY = floor(y);
	float fractY = fract(y);

	float v1 = random(vec2(intX, intY), vec2(0));
	float v2 = random(vec2(intX + 1.0f, intY), vec2(0));
	float v3 = random(vec2(intX, intY + 1.0f), vec2(0));
	float v4 = random(vec2(intX + 1.0f, intY + 1.0f), vec2(0));

	float i1 = mix(v1, v2, fractX);
	float i2 = mix(v3, v4, fractX);
	return mix(i1, i2, fractY);
}

float fbm2(vec2 p) {
	float total = 0.0f;
	float persistence = 0.5f;
	int octaves = 8;

	for(int i = 0; i < octaves; i++) {
		float freq = pow(2.0f, float(i));
		float amp = pow(persistence, float(i));
		total += interpNoise2D(p.x * freq, p.y * freq) * amp;
	}

	return total;
}

float squareWave(float x, float freq, float amplitude) {
	return abs(float(int(floor(x * freq)) % 2) * amplitude);
}

float sawtoothWave(float x, float freq, float amplitude) {
	return (x * freq - floor(x * freq)) * amplitude;
}

#define cell_size 3.

vec2 generate_point(vec2 cell) {
    vec2 p = vec2(cell.x, cell.y);
    p += fract(sin(vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)) * 43758.5453)));
    return p * cell_size;
}

float worleyNoise(vec2 pixel) {
    vec2 cell = floor(pixel / cell_size);

    vec2 point = generate_point(cell);

    float shortest_distance = length(pixel - point);

   // compute shortest distance from cell + neighboring cell points

    for(float i = -1.0f; i <= 1.0f; i += 1.0f) {
        float ncell_x = cell.x + i;
        for(float j = -1.0f; j <= 1.0f; j += 1.0f) {
            float ncell_y = cell.y + j;

            // get the point for that cell
            vec2 npoint = generate_point(vec2(ncell_x, ncell_y));

            // compare to previous distances
            float distance = length(pixel - npoint);
            if(distance < shortest_distance) {
                shortest_distance = distance;
            }
        }
    }

    return shortest_distance / cell_size;
}

vec2 worleyPoint(vec2 pixel) {
	vec2 cell = floor(pixel / cell_size);

    vec2 point = generate_point(cell);

    float shortest_distance = length(pixel - point);

   // compute shortest distance from cell + neighboring cell points

    for(float i = -1.0f; i <= 1.0f; i += 1.0f) {
        float ncell_x = cell.x + i;
        for(float j = -1.0f; j <= 1.0f; j += 1.0f) {
            float ncell_y = cell.y + j;

            // get the point for that cell
            vec2 npoint = generate_point(vec2(ncell_x, ncell_y));

            // compare to previous distances
            float distance = length(pixel - npoint);
            if(distance < shortest_distance) {
                shortest_distance = distance;
                point = npoint;
            }
        }
    }

    return point;
}

/*
 * MY SDFS
 */

float wallSDF(vec3 p) {
	return boxSDF(p, vec3(22.0, 12.5, 0.1));
}

float backWallSDF(vec3 p) {
	vec3 q = translate(p, 0.0, 0.0, -25.0);
	return opSubtraction(boxSDF(translate(q, 0., 3.3, 0.), vec3(7.5, 9., 2.)), wallSDF(q));
}

float leftWallSDF(vec3 p) {
	vec3 q = translate(rotateY(p, 90.0), 10.0, 0.0, -21.0);
	return opSubtraction(boxSDF(translate(q, -4.5, 2., 0.0), vec3(8., 5., 1.)),
					wallSDF(q));
}

float rightWallSDF(vec3 p) {
	vec3 q = translate(rotateY(p, -90.0), -10.0, 0.0, -21.0);
	return opSubtraction(boxSDF(translate(q, 0., 3., -1.0), vec3(4.0, 10., 2.)),
					wallSDF(q));
}

float ceilingSDF(vec3 p) {
	return boxSDF(translate(p, 0., -10.2, -3.), vec3(22.0, 0.2, 22.0));
}

float doorSDF(vec3 p) {
    float c = cos(radians(.4 * p.y));
    float s = sin(radians(.4 * p.y));
    mat2  m = mat2(c, -s, s, c);
    vec3  q = translate(vec3(m * p.xy, p.z), -0.7, 0.8, 0.);
    float t;
    int time = int(u_Time) % 110;
    if(time > 30) {
    	t = 0.;
    } else {
    	t = abs(sawtoothWave(float(time) * 0.4, 0.4, 1.0));
    }
	return boxSDF(translate(mix(p, q, t), 20., 2.55, -10.),
						    mix(vec3(0.4, 9.4, 3.45), vec3(0.4, 11.2, 3.3), t));
}

float doorframeSDF(vec3 p) {
	vec3 q = translate(p, 20.5, 2.4, -10.0);
	return opSubtraction(boxSDF(translate(q, 0., 0.37, 0.), vec3(0.9, 9.8, 3.5)),
						 boxSDF(q, vec3(0.7, 10.0, 4.0)));
}

float windowframeSDF(vec3 p) {
	vec3 q = translate(p, -20.5, 1.8, -5.25);
	return opSubtraction(boxSDF(q, vec3(1.0, 5., 8.)), boxSDF(q, vec3(0.4, 5.5, 8.25)));
}

float closetframeSDF(vec3 p) {
	vec3 q = translate(p, 0., 3.75, -25.);
	return opSubtraction(boxSDF(q, vec3(7.5, 9., 2.)), boxSDF(q, vec3(8., 9.5, .7)));
}

float roomSDF(vec3 p) {
	return opUnion( opUnion( opUnion(
							 	opSmoothUnion(rightWallSDF(p), doorframeSDF(p), .2),
							 	opSmoothUnion(backWallSDF(p), closetframeSDF(p), .2)),
							 	opSmoothUnion(leftWallSDF(p), windowframeSDF(p), .2)),
							  ceilingSDF(p));
}

float bedsheetSDF(vec3 p) {
    float c = cos(radians(-1. * p.x));
    float s = sin(radians(-1. * p.x));
    mat2  m = mat2(c, -s, s, c);
    vec3  q = vec3(m * p.xy, p.z);
    float bump = worleyNoise(p.xz);
	return boxSDF(translate(q, 0.0, 4., 4.5), vec3(6.2, 1.0 + 0.4 * cos(bump), 4.0));
}

float bedcoverSDF(vec3 p) {
	float c = cos(radians(-1. * p.x));
    float s = sin(radians(-1. * p.x));
    mat2  m = mat2(c, -s, s, c);
    vec3  q = vec3(m * p.xy, p.z);
    float bump = cos(p.x + p.y + p.z);
	return boxSDF(translate(q, 0.0, 2.5, 5.5), vec3(6.2, .25 + 0.1 * bump, 1.0));
}

float postSDF(vec3 p) {	  	 
	return opUnion(cappedCylinderSDF(p, vec2(0.4, 2.0)),
		   opSmoothUnion(cappedCylinderSDF(translate(p, 0.0, -2.0, 0.0), vec2(0.7, 0.2)),
		   		   			     sphereSDF(translate(p, 0.0, -2.4, 0.0), 0.57), 0.15));
}

float bedpostsSDF(vec3 p) {
    p.x = abs(p.x);
    return postSDF(vec3(translate(p, -6.5, 4.2, 0.9)));
}

float bedSDF(vec3 p) {
	return opUnion(bedpostsSDF(p), opUnion(bedcoverSDF(p), bedsheetSDF(p)));
}

float lampshadeSDF(vec3 p) {
	vec3 q = translate(p, 0.0, -10.0, 0.0);
	vec3 lamp = translate(q, 13.0, 9.5, -17.0);
	return opSubtraction(cappedConeSDF(lamp, 1.5, 1.7, 1.1),
						 cappedConeSDF(lamp, 1.3, 1.9, 1.3));
}

float lamppostSDF(vec3 p) {
	vec3 lamp = translate(p, 13.0, 9.5, -17.0);
	return opSmoothUnion(cappedConeSDF(lamp, 0.15, 1.2, 0.7),
					 	 cappedCylinderSDF(translate(lamp, 0., -5.1, 0.), vec2(0.1, 5.5)), 1.0);
}

float lampSDF(vec3 p) {
	return opUnion(lampshadeSDF(p), lamppostSDF(p));
}

float eyebaseSDF(vec3 p) {
	p.x = abs(p.x);
	vec3 q = translate(rotateZ(p, 30.), -1.5, 0., 0.);
	return opIntersection(sphereSDF(q, 0.6),
					 	  boxSDF(translate(q, 0., 0.6, 0.), vec3(3., 0.6, 3.)));
}

float eyesBackSDF(vec3 p) {
	vec3 q1 = translate(rotateX(p, -30.), -2., 2., -25.);
	//vec3 q2 = translate(p, )
	return eyebaseSDF(q1);
}

/*
float eyesLeftSDF(vec3 p) {
	
} */

float sceneSDF(vec3 p) {
	return opUnion( opUnion( opUnion( opUnion( bedSDF(p),
		   		   			roomSDF(p)),
		   		   			lampSDF(p)),
		   		   			doorSDF(p)),
		   		   			eyesBackSDF(p));

} 

/*
 * RAY MARCHING
 */

vec3 rayCast() {
  vec3 forward = normalize(u_Ref - u_Eye);
  vec3 right = cross(forward, u_Up);
  vec3 up = cross(right, forward);
  float len = length(u_Ref - u_Eye);

  vec3 v = up * len * tan(22.5f);
  vec3 h = right * len * (u_Dimensions.x / u_Dimensions.y) * tan(22.5f);
  vec3 p = u_Ref + fs_Pos.x * h + fs_Pos.y * v;
  return normalize(p - u_Eye);
}

bool intersectsBox(vec3 dir, vec3 min, vec3 max) {
	float tmin = (min.x - u_Eye.x) / dir.x;
	float tmax = (max.x - u_Eye.x) / dir.x;

	if (tmin > tmax) {
		float temp = tmin;
		tmin = tmax;
		tmax = temp;
	}

	float tymin = (min.y - u_Eye.y) / dir.y;
	float tymax = (max.y - u_Eye.y) / dir.y;

	if (tymin > tymax) {
		float temp = tymin;
		tymin = tymax;
		tymax = temp;
	}

	if ((tmin > tymax) || (tymin > tmax)) {
		return false; 
	}
 
    if (tymin > tmin) {
    	tmin = tymin;
    }
 
    if (tymax < tmax) {
    	tmax = tymax; 
    }
 
    float tzmin = (min.z - u_Eye.z) / dir.z; 
    float tzmax = (max.z - u_Eye.z) / dir.z; 
 
    if (tzmin > tzmax) {
		float temp = tzmin;
		tzmin = tzmax;
		tzmax = temp;
    }
 
    if ((tmin > tzmax) || (tzmin > tmax)) {
        return false; 
    }

	return true;
}

bool withinBed(vec3 dir) {
	return intersectsBox(dir, vec3(-9.0, -5.0, -5.5), vec3(9.0, 4.0, 10.0));
}

bool withinBackEye(vec3 dir) {
	return intersectsBox(dir, vec3(-10, -10.0, 10.), vec3(20.0, 20., 30.));
}

// March along the ray
#define max_steps 200	
#define cutoff 50.0f
void march(vec3 origin, vec3 direction) {

	float t = 0.0f;
	vec3 pos;
	bool interBed = withinBed(direction);
	bool interEyesB = withinBackEye(direction);

	for(int i = 0; i < max_steps; i++) {
		pos = origin + t * direction;/*
		float dist = roomSDF(pos);

		if(interBed) {
			dist = opUnion(dist, bedSDF(pos));
		}

		dist = opUnion(dist, doorSDF(pos));
		dist = opUnion(dist, lampSDF(pos));
		dist = opUnion(dist, eyesSDF(pos));*/

		if(!interEyesB) {
			return;
		}

		float dist = eyesBackSDF(pos);

		if(dist < 0.05) {
			if(floatEquality(dist, roomSDF(pos))) {
		    	color_Id = WALLS;
		    } else if(floatEquality(dist, bedpostsSDF(pos))) {
		    	color_Id = WOOD;
		    } else if(floatEquality(dist, bedsheetSDF(pos))) {
		    	color_Id = BED;
		    } else if(floatEquality(dist, bedcoverSDF(pos))) {
		    	color_Id = BEDCOVER;
		   	} else if(floatEquality(dist, lamppostSDF(pos))) {
		    	color_Id = LAMPPOST;
		    } else if(floatEquality(dist, lampshadeSDF(pos))) {
		    	color_Id = LAMPSHADE;
		    } else if(floatEquality(dist, doorSDF(pos))) {
		    	color_Id = DOOR;
		    } else if(floatEquality(dist, eyesBackSDF(pos))) {
		    	color_Id = EYES;
		    }

		    final_Pos = pos;
			return;
		}

		t += dist;

		if(t >= cutoff) {
			color_Id = BACKGROUND;
			return;
		}
	}
}

#define epsilon 0.0005f
vec3 getNormal(vec3 p) {
	return normalize(vec3(sceneSDF(vec3(p.x + epsilon, p.y, p.z))
						- sceneSDF(vec3(p.x - epsilon, p.y, p.z)),
						  sceneSDF(vec3(p.x, p.y + epsilon, p.z))
					    - sceneSDF(vec3(p.x, p.y - epsilon, p.z)),
					      sceneSDF(vec3(p.x, p.y, p.z + epsilon))
					    - sceneSDF(vec3(p.x, p.y, p.z - epsilon))));
}

float softshadow(vec3 p, float k)
{
	vec3 ro = point_Light;
	vec3 rd = normalize(p - point_Light);
    float res = 1.0;
    for(float t = 0.1; t < 45.; ) {
        float h = sceneSDF(ro + rd * t);
        if( h < 0.001 ) {
        	return 0.;
        }
           
        res = min(res, k * h / t );
        t += h;
    }

    return res;
}

vec4 applyLambertReg(vec3 p, vec3 n, vec3 base, vec3 shadowColor) {
	float lambert = clamp(dot(n, normalize(light_Vec)), 0.0, 1.0);
	float lambert2 = clamp(dot(n, normalize(light_Vec2)), 0.0, 1.0);
	// Add ambient lighting
	float ambientTerm = 0.2;
	vec3 lightIntensity = (light_Vec_Col * (lambert + ambientTerm) +
						  light_Vec2_Col * (lambert2 + ambientTerm)) / 2.;
	vec3 shadow = 0.3 * shadowColor * (1.0 - lightIntensity);
	return vec4(clamp(base * lightIntensity + shadow, 0.0f, 1.0f), 1.0f);
}

vec4 applyLambert(vec3 p, vec3 base, vec3 shadowColor) {
	vec3 n = (getNormal(p));
	float lambert = clamp(dot(n, normalize(light_Vec)), 0.0, 1.0);
	float lambert2 = clamp(dot(n, normalize(light_Vec2)), 0.0, 1.0);
	float lambert3 = clamp(dot(n, normalize(point_Light - p)), 0.0, 1.0);
	// Add ambient lighting
	float ambientTerm = 0.4;
	vec3 lightIntensity = ((light_Vec_Col * lambert) + (light_Vec2_Col * lambert2)
						  + (point_Light_Col * lambert3)) / 3. + ambientTerm;
	return vec4(clamp(base * lightIntensity, 0.0f, 1.0f), 1.0f);
}

vec4 applyBlinnPhong(vec3 p, vec3 base, float power, vec3 shadowColor) {
	vec3 n = (getNormal(p));
	vec4 lambert = applyLambert(p, base, shadowColor);
	// Average the view / light vector
    vec3 h_vector = (normalize(light_Vec) + normalize(u_Eye)) / 2.0f;
	// Calculate specular intensity
	float specularIntensity = max(pow(dot(h_vector, n), power), 0.0);
	return vec4(clamp(vec3(lambert.rgb + .2 * specularIntensity), 0.0f, 1.0f), 1.0f);
}

vec4 getColor(vec3 p) {
	int iTime = int(u_Time) % 60;
	float time;
	if(iTime > 30) {
		time = 0.;
	} else {
		time = squareWave(u_Time, 0.5, 1.0);
	}

	point_Light_Col = mix(point_Light_Base, vec3(0.), time);

	// Wood cosine palette
	vec3 a = vec3(0.71, 0.72, 0.0);
	vec3 b = vec3(0.56, 0.68, 1.0);
	vec3 c = vec3(0.70, 0.41, 0.36);
	vec3 d = vec3(0.00, 0.23, 0.36);


	vec3 normal = (getNormal(p));
	switch(color_Id) {
		case WALLS:
			return applyLambert(p, vec3(0.7), vec3(0.));
		case WOOD:
			vec3 wood = vec3(96.0, 77.0, 54.0) / 255.0;
			float t = fract(fbm(p.x * 15.0));
			return applyBlinnPhong(p, mix(a + b * cos(6.28318531 * (c * t + d)),
									   wood, 1. - 0.6 * t), 6., vec3(0.));
		case DOOR:
			vec3 door = vec3(73., 33., 8.) / 255.0;
			float t2 = fract(fbm(p.z * 10.0));
			return applyLambert(p, mix(a + b * cos(6.28318531 * (c * t2 + d)),
									   door, 1. - 0.6 * t2), vec3(0.));
		case BED:
			vec3 bed = vec3(91.0, 2.0, 2.0) / 255.0;
			return applyBlinnPhong(p, bed, 2., vec3(0.));
		case BEDCOVER:
			return applyLambert(p, vec3(1.), vec3(0.));
		case LAMPPOST:
			return applyBlinnPhong(p, vec3(0.2), 5., vec3(0.));
		case LAMPSHADE:
			#define DISTORTION 0.9f
			#define GLOW 3.0
			#define SCALE 1.8;

			vec3 lamp_col = vec3(211., 198., 131.) / 255.;
			vec3 light = normalize(point_Light - p);
			vec3 view = normalize(p - u_Eye);
			vec3 normal = getNormal(p);
			vec3 scatterDir = (point_Light - p) + normal * DISTORTION;
			float lightReachingEye = pow(clamp(dot(view, -scatterDir),
												0., 1.), GLOW) * SCALE;
			float attenuation = max(0., dot(normal, light) +
										dot(view, -light));
			float totalLight = attenuation + (lightReachingEye + 0.2) * 0.9;
			return applyLambertReg(p, normal, lamp_col * point_Light_Col * totalLight, vec3(0.));
		case EYES:
			return vec4(vec3(209., 24., 14.) / 255., 1.);
		case BACKGROUND:
		default:
			return vec4(0., 0., 0., 1.);
	}
}

void main() {
  vec3 dir = rayCast();
  march(u_Eye, dir);

  out_Col = getColor(final_Pos);
  out_Col *= 1.6 + atan(8. * out_Col - 5.5);
  out_Col.b *= 4.;
  // vignette
  vec2 distance_center = fs_Pos / 2.;
  out_Col *= clamp(1. - length(distance_center), 0., 1.);
  out_Col.a = 1.;
}