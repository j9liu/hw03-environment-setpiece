#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;

in vec2 fs_Pos;
out vec4 out_Col;

int color_Id = -1;

// Define color/shader IDs.
#define BACKGROUND 0

bool floatEquality(float x, float y) {
	return abs(x - y) < 0.0001;
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

vec2 opRevolution(vec3 p, float w) {
    return vec2( length(p.xz) - w, p.y );
}

/*
 * MY SDFS
 */

float sceneSDF(vec3 p) {
	return opUnion(spoonSDF(p), opUnion(bowlSDF(p), opUnion(marbleScoopSDF(p), mintScoopSDF(p))));
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

float perturbedFbm(vec2 p) {
      vec2 q = vec2(fbm2(p + vec2(0.0,0.0)),
                    fbm2(p + vec2(5.2,1.3)));
      vec2 r = vec2(fbm2(p + 4.0*q + vec2(9.7,9.2)),
                    fbm2(p + 4.0*q + vec2(8.3,2.8)));
      return fbm2( p + 4.0*r );
}

float bias (float b, float t) {
	return pow(t, log(b) / log(0.5f));
}

float gain (float g, float t) {
	if(t < 0.5f) {
		return bias(1.0 - g, 2.0 * t) / 2.0;
	}
	return 1.0 - bias(1.0 - g, 2.0 - 2.0 * t) / 2.0;
}


#define cell_size 0.3

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

#define epsilon 0.0005f
vec3 getNormal(vec3 p) {
	return normalize(vec3(sceneSDF(vec3(p.x + epsilon, p.y, p.z))
						- sceneSDF(vec3(p.x - epsilon, p.y, p.z)),
						  sceneSDF(vec3(p.x, p.y + epsilon, p.z))
					    - sceneSDF(vec3(p.x, p.y - epsilon, p.z)),
					      sceneSDF(vec3(p.x, p.y, p.z + epsilon))
					    - sceneSDF(vec3(p.x, p.y, p.z - epsilon))));
}

vec4 applyLambert(vec3 p, vec3 base, vec3 shadowColor) {
	vec3 normal = getNormal(p);
	float lambert = clamp(dot(normalize(normal), normalize(light_Vec)), 0.0, 1.0);
	// Add ambient lighting
	float ambientTerm = 0.2;
	float lightIntensity = lambert + ambientTerm;
	vec3 shadow = 0.3 * shadowColor * (1.0 - lightIntensity);
	return vec4(clamp(base * lightIntensity + shadow, 0.0f, 1.0f), 1.0f);
}

vec4 applyBlinnPhong(vec3 p, vec3 base, float power, vec3 shadowColor) {
	vec3 normal = getNormal(p);
	vec4 lambert = applyLambert(p, base, shadowColor);
	// Average the view / light vector
    vec3 h_vector = (normalize(light_Vec) + normalize(u_Eye)) / 2.0f;
	// Calculate specular intensity
	float specularIntensity = max(pow(dot(h_vector, normal), power), 0.0);
	return vec4(clamp(vec3(lambert.rgb + .2 * specularIntensity), 0.0f, 1.0f), 1.0f);
}

vec4 getColor(vec3 p) {
	switch(color_Id) {
		case BOWL:
			vec3 color = vec3(196.0, 25.0, 82.0) / 255.0;
			float stripe = pow(fbm(2.0f * fs_Pos.x + fs_Pos.y + gain(0.6, 0.01 * float(int(u_Time) % 2000))), 5.0f);
			if(stripe > 1.0f) {
				color += vec3(0.6, 0.6, 0.8) * abs(fs_Pos.y);
			}
			return applyBlinnPhong(p, color, 4.0f, vec3(76.0, 15.0, 52.0) / 255.0f);
		case MARBLED:
			vec3 gray = vec3(20.0, 20.0, 25.0) / 255.0;
			vec3 marble = clamp(gray + vec3(pow(perturbedFbm(1.489 * sin(p.xz) + p.zy), 3.0f)), 0.0, 1.0);
			marble = clamp((1.0 - marble) + u_MarbleBase, 0.0, 1.0);
			marble *= perturbedFbm(.00000232 * cos(p.xz * p.xy) + sin(p.yz));
			return applyLambert(p, marble, vec3(76.0, 15.0, 52.0) / 255.0f);
		case MINT:
			vec3 mint = u_SolidBase;
			mint *= perturbedFbm(.00009832 * cos(p.xz * p.xy) - sin(p.yz));
			vec2 chocchip = worleyPoint(vec2(p.z, p.x + p.y));
			if(noise(chocchip.x + chocchip.y) > 0.97) {
				mint = u_ChipColor;
			}
			return applyLambert(p, mint, 1.1 * vec3(76.0, 15.0, 52.0) / 255.0f);
		case SPOON:
			vec3 silver = vec3(101, 106, 107) / 255.0;
			return applyBlinnPhong(p, silver, 6.0f, vec3(76.0, 15.0, 52.0) / 255.0f);
		case BACKGROUND:
		default:
			return vec4(mix(vec3(255.0, 247.0, 226.0) / 255.0,
							vec3(206.0, 226.0, 226.0) / 255.0,
							vec3(fs_Pos.xy, fs_Pos.x * fs_Pos.y) / 2.0), 1.0);
	}
}

// March along the ray
#define max_steps 200	
#define cutoff 1100.0f
void march(vec3 direction) {
	// check if will hit; if it doesn't, return early
	if(!withinVolume(direction)) {
		color_Id = BACKGROUND;
		out_Col = getColor(vec3(0));
		return;
	}

	float t = 0.0f;
	int temp = 1;
	vec3 pos;

	for(int i = 0; i < max_steps; i++) {
		pos = u_Eye + t * direction;
		float dist = sceneSDF(pos);
		if(dist < 0.015) {
			return;
		}

		t += dist;

		if(t >= cutoff) {
			out_Col = getColor(pos);
			return;
		}
	}
}

void main() {
  vec3 dir = rayCast();
  march(dir);
}
