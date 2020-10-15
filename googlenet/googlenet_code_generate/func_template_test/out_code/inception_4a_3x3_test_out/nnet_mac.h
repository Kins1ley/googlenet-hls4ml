#ifndef NNET_MAC_H_
#define NNET_MAC_H_

#include "nnet_common.h"
#include <cstdlib>

namespace nnet {

template<typename CONFIG_T>
typename CONFIG_T::out_t  mac3(
	typename  CONFIG_T::in_t  b0, typename CONFIG_T::weight_t w0,
	typename CONFIG_T::in_t  b1, typename CONFIG_T::weight_t w1,
	typename CONFIG_T::in_t  b2,typename CONFIG_T::weight_t w2)
{
	typename CONFIG_T::accum_t  mul0, mul1, mul2;
	typename CONFIG_T::accum_t  add00;
	typename CONFIG_T::accum_t  add1;

	mul0 = w0 * b0;
	mul1 = w1 * b1;
	mul2 = w2 * b2;

	add00 = mul0 + mul1;
	add1 = add00 + mul2;

	return add1;

}

template<typename CONFIG_T>
typename CONFIG_T::out_t  mac5(
	typename  CONFIG_T::in_t  b0, typename CONFIG_T::weight_t w0,
	typename CONFIG_T::in_t  b1, typename CONFIG_T::weight_t w1,
	typename CONFIG_T::in_t  b2, typename CONFIG_T::weight_t w2,
	typename CONFIG_T::in_t  b3, typename CONFIG_T::weight_t w3,
	typename CONFIG_T::in_t  b4, typename CONFIG_T::weight_t w4)
{
	typename CONFIG_T::accum_t  mul0, mul1, mul2, mul3, mul4;
	typename CONFIG_T::accum_t  add00, add01;
	typename CONFIG_T::accum_t  add10;
	typename CONFIG_T::accum_t  add2;
	mul0 = w0 * b0;
	mul1 = w1 * b1;
	mul2 = w2 * b2;
	mul3 = w3 * b3;
	mul4 = w4 * b4;

	add00 = mul0 + mul1;
	add01 = mul2 + mul3;

	add10 = add00 + add01;

	add2 = add10 + mul4;

	return add2;

}

template<typename CONFIG_T>
typename CONFIG_T::out_t  sum5(
	typename  CONFIG_T::in_t  b0,
	typename CONFIG_T::in_t  b1,
	typename CONFIG_T::in_t  b2,
	typename CONFIG_T::in_t  b3,
	typename CONFIG_T::in_t  b4)
{

	typename CONFIG_T::accum_t  add00, add01;
	typename CONFIG_T::accum_t  add10;
	typename CONFIG_T::accum_t  add2;


	add00 = b0 + b1;
	add01 = b2 + b3;


	add10 = add00 + add01;

	add2 = add10 + b4;

	return add2;

}
template<typename CONFIG_T>
typename CONFIG_T::accum_t  mac7(
	typename  CONFIG_T::in_t  b0, typename CONFIG_T::weight_t w0,
	typename CONFIG_T::in_t  b1, typename CONFIG_T::weight_t w1,
	typename CONFIG_T::in_t  b2, typename CONFIG_T::weight_t w2,
	typename CONFIG_T::in_t  b3, typename CONFIG_T::weight_t w3,
	typename CONFIG_T::in_t  b4, typename CONFIG_T::weight_t w4,
	typename CONFIG_T::in_t  b5, typename CONFIG_T::weight_t w5,
	typename CONFIG_T::in_t  b6,typename CONFIG_T::weight_t w6 )
{
	typename CONFIG_T::accum_t  mul0, mul1, mul2, mul3, mul4, mul5, mul6;
	typename CONFIG_T::accum_t  add00, add01, add02;
	typename CONFIG_T::accum_t  add10, add11;
	typename CONFIG_T::accum_t  add2;

	mul0 = w0 * b0;
	mul1 = w1 * b1;
	mul2 = w2 * b2;
	mul3 = w3 * b3;
	mul4 = w4 * b4;
	mul5 = w5 * b5;
	mul6 = w6 * b6;


	add00 = mul0 + mul1;
	add01 = mul2 + mul3;
	add02 = mul4 + mul5;

	add10 = add00 + add01;
	add11 = add02 + mul6;

	add2 = add10 + add11;

	return add2;

}

template<typename CONFIG_T>
typename CONFIG_T::out_t  sum7(
	typename CONFIG_T::accum_t  b0,
	typename CONFIG_T::accum_t  b1,
	typename CONFIG_T::accum_t  b2,
	typename CONFIG_T::accum_t  b3,
	typename CONFIG_T::accum_t  b4,
	typename CONFIG_T::accum_t  b5,
	typename CONFIG_T::accum_t  b6)
{

	typename CONFIG_T::accum_t  add00, add01, add02;
	typename CONFIG_T::accum_t  add10, add11;
	typename CONFIG_T::accum_t  add2;


	add00 = b0 + b1;
	add01 = b2 + b3;
	add02 = b4 + b5;

	add10 = add00 + add01;
	add11 = add02 + b6;

	add2 = add10 + add11;

	return add2;

}

template<typename CONFIG_T>
typename CONFIG_T::out_t  mac9(
	typename  CONFIG_T::in_t  b0, typename CONFIG_T::weight_t w0,
	typename CONFIG_T::in_t  b1, typename CONFIG_T::weight_t w1,
	typename CONFIG_T::in_t  b2, typename CONFIG_T::weight_t w2,
	typename CONFIG_T::in_t  b3, typename CONFIG_T::weight_t w3,
	typename CONFIG_T::in_t  b4, typename CONFIG_T::weight_t w4,
	typename CONFIG_T::in_t  b5, typename CONFIG_T::weight_t w5,
	typename CONFIG_T::in_t  b6, typename CONFIG_T::weight_t w6,
	typename CONFIG_T::in_t  b7, typename CONFIG_T::weight_t w7,
	typename CONFIG_T::in_t  b8,typename CONFIG_T::weight_t w8)
{
	typename CONFIG_T::accum_t  mul0, mul1, mul2, mul3, mul4, mul5, mul6, mul7, mul8;
	typename CONFIG_T::accum_t  add00, add01, add02, add03;
	typename CONFIG_T::accum_t  add10, add11;
	typename CONFIG_T::accum_t  add20;
	typename CONFIG_T::accum_t  add3;

	mul0 = w0 * b0;
	mul1 = w1 * b1;
	mul2 = w2 * b2;
	mul3 = w3 * b3;
	mul4 = w4 * b4;
	mul5 = w5 * b5;
	mul6 = w6 * b6;
	mul7 = w7 * b7;
	mul8 = w8 * b8;

	add00 = mul0 + mul1;
	add01 = mul2 + mul3;
	add02 = mul4 + mul5;
	add03 = mul6 + mul7;

	add10 = add00 + add01;
	add11 = add02 + add03;

	add20 = add10 + add11;

	add3 = add20 + mul8;

	return add3;

}

template<typename CONFIG_T>
typename CONFIG_T::out_t  mac25(
	typename  CONFIG_T::in_t  b0, typename CONFIG_T::weight_t w0,
	typename  CONFIG_T::in_t  b1, typename CONFIG_T::weight_t w1,
	typename  CONFIG_T::in_t  b2, typename CONFIG_T::weight_t w2,
	typename  CONFIG_T::in_t  b3, typename CONFIG_T::weight_t w3,
	typename  CONFIG_T::in_t  b4, typename CONFIG_T::weight_t w4,
	typename  CONFIG_T::in_t  b5, typename CONFIG_T::weight_t w5,
	typename  CONFIG_T::in_t  b6, typename CONFIG_T::weight_t w6,
	typename  CONFIG_T::in_t  b7, typename CONFIG_T::weight_t w7,
	typename  CONFIG_T::in_t  b8, typename CONFIG_T::weight_t w8,
	typename  CONFIG_T::in_t  b9, typename CONFIG_T::weight_t w9,
	typename  CONFIG_T::in_t  b10, typename CONFIG_T::weight_t w10,
	typename  CONFIG_T::in_t  b11, typename CONFIG_T::weight_t w11,
	typename  CONFIG_T::in_t  b12, typename CONFIG_T::weight_t w12,
	typename  CONFIG_T::in_t  b13, typename CONFIG_T::weight_t w13,
	typename  CONFIG_T::in_t  b14, typename CONFIG_T::weight_t w14,
	typename  CONFIG_T::in_t  b15, typename CONFIG_T::weight_t w15,
	typename  CONFIG_T::in_t  b16, typename CONFIG_T::weight_t w16,
	typename  CONFIG_T::in_t  b17, typename CONFIG_T::weight_t w17,
	typename  CONFIG_T::in_t  b18, typename CONFIG_T::weight_t w18,
	typename  CONFIG_T::in_t  b19, typename CONFIG_T::weight_t w19,
	typename  CONFIG_T::in_t  b20, typename CONFIG_T::weight_t w20,
	typename  CONFIG_T::in_t  b21, typename CONFIG_T::weight_t w21,
	typename  CONFIG_T::in_t  b22, typename CONFIG_T::weight_t w22,
	typename  CONFIG_T::in_t  b23, typename CONFIG_T::weight_t w23,
	typename  CONFIG_T::in_t  b24, typename CONFIG_T::weight_t w24 )
{
	typename CONFIG_T::accum_t  mul0, mul1, mul2, mul3, mul4, mul5, mul6, mul7, mul8, mul9, mul10, mul11, mul12, mul13, mul14, mul15, mul16, mul17, mul18, mul19, mul20, mul21, mul22, mul23, mul24;
	typename CONFIG_T::accum_t  add00, add01, add02, add03, add04, add05, add06, add07, add08, add09, add010, add011, add012, add013;
	typename CONFIG_T::accum_t  add10, add11, add12, add13, add14, add15, add16, add17, add18, add19, add110, add111, add112;
	typename CONFIG_T::accum_t  add20,add21,add22,add23,add24;
	typename CONFIG_T::accum_t  add30,add31;
	typename CONFIG_T::accum_t  add4;

	mul0 = w0 * b0;
	mul1 = w1 * b1;
	mul2 = w2 * b2;
	mul3 = w3 * b3;
	mul4 = w4 * b4;
	mul5 = w5 * b5;
	mul6 = w6 * b6;
	mul7 = w7 * b7;
	mul8 = w8 * b8;
	mul9 = w9 * b9;
	mul10 = w10 * b10;
	mul11 = w11 * b11;
	mul12 = w12 * b12;
	mul13 = w13 * b13;
	mul14 = w14 * b14;
	mul15 = w15 * b15;
	mul16 = w16 * b16;
	mul17 = w17 * b17;
	mul18 = w18 * b18;
	mul19 = w19 * b19;
	mul20 = w20 * b20;
	mul21 = w21 * b21;
	mul22 = w22 * b22;
	mul23 = w23 * b23;
	mul24 = w24 * b24;

	add00 = mul0 + mul1;
	add01 = mul2 + mul3;
	add02 = mul4 + mul5;
	add03 = mul6 + mul7;
	add04 = mul8 + mul9;
	add05 = mul10 + mul11;
	add06 = mul12 + mul13;
	add07 = mul14 + mul15;
	add08 = mul16 + mul17;
	add09 = mul18 + mul19;
	add010 = mul20 + mul21;
	add011 = mul22 + mul23;

	add10 = add00 + add01;
	add11 = add02 + add03;
	add12 = add04 + add05;
	add13 = add06 + add07;
	add14 = add08 + add09;
	add15 = add010 + add011;

	add20 = add10 + add11;
	add21 = add12 + add13;
	add22 = add14 + add15;

	add30 = add20 + add21;
	add31 = add22 + mul24;

	add4 = add30 + add31;
	return add4;

}


} // end namespace

#endif