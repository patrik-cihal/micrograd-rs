use micrograd::*;

fn main() {
    let x1 = Value::new(2.);
    let x2 = Value::new(0.);
    let w1 = Value::new(-3.);
    let w2 = Value::new(1.);

    let b = Value::new(6.881373587019);

    let x1w1 = x1.c()*w1.c();
    let x2w2 = x2.c()*w2.c();
    
    let x1w1x2w2 = x1w1+x2w2;
    let n = x1w1x2w2+b;
    let o = n.c().tanh();

    dbg!(o.eval());

    o.compute_gradient("loss");

    dbg!(n.grad("loss"), x1.grad("loss"), x2.grad("loss"), w1.grad("loss"), w2.grad("loss"), o.grad("loss"));
}
