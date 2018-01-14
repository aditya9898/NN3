function sig=sigmoid(x)
  a=exp(-x);
  a=1+a;
  a=1./a;
  sig=a;
  end
  