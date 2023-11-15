use std::{ops::{Add, Mul, Sub}, collections::{BTreeSet, HashMap}, rc::Rc, cell::RefCell};

enum Operation {
    Mul(Value, Value),
    Add(Value, Value),
    Init,
    Exp(Value),
    Powf(Value, f64)
}


struct ValueInner {
    operation: Operation,
    eval: f64,
    grad: HashMap<String, f64>
}

impl ValueInner {
    fn propagate_gradient(&mut self, label: &str, cur: f64) {
        *self.grad.entry(label.to_string()).or_insert(0.) += cur;
        match &self.operation {
            Operation::Mul(val1, val2) => {
                let eval1 = val1.eval();
                let eval2 = val2.eval();
                val1.value_inner.borrow_mut().propagate_gradient(label, cur*eval2);
                val2.value_inner.borrow_mut().propagate_gradient(label, cur*eval1);
            },
            Operation::Add(val1, val2) => {
                val1.value_inner.borrow_mut().propagate_gradient(label, cur);
                val2.value_inner.borrow_mut().propagate_gradient(label, cur);
            },
            Operation::Exp(val) => {
                let meval = self.eval;
                val.value_inner.borrow_mut().propagate_gradient(label, cur*meval);
            }
            
            Operation::Init => (),
            Operation::Powf(val, p) => {
                let d = val.eval().powf(p-1.)*(*p);
                val.value_inner.borrow_mut().propagate_gradient(label, cur*d);
            },
        }
    }
}

#[derive(Clone)]
pub struct Value {
    value_inner: Rc<RefCell<ValueInner>>
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

impl Value {
    pub fn new(data: f64) -> Self {
        let value_inner = ValueInner {
            eval: data,
            operation: Operation::Init,
            grad: Default::default()
        };
        Self {
            value_inner: Rc::new(RefCell::new(value_inner))
        }
    }
    pub fn c(&self) -> Self {
        self.clone()
    }
    pub fn compute_gradient(&self, label: &str) {
        self.value_inner.borrow_mut().propagate_gradient(label, 1.);
    }
    pub fn eval(&self) -> f64 {
        self.value_inner.borrow().eval
    }
    pub fn grad(&self, label: &str) -> f64 {
        self.value_inner.borrow().grad[label]
    }
    pub fn tanh(self) -> Value {
        let e2x = (self.c()*2.0.into()).exp();
        (e2x.c()-1.0.into())*(e2x+1.0.into()).powf(-1.)
    }
    pub fn exp(self) -> Value {
        let value_inner = ValueInner {
            eval: self.eval().exp(),
            operation: Operation::Exp(self),
            grad: Default::default()
        };

        Value {
            value_inner: Rc::new(RefCell::new(value_inner))
        } 
    }
    pub fn powf(self, p: f64) -> Value {
        let value_inner = ValueInner {
            eval: self.eval().powf(p),
            operation: Operation::Powf(self, p),
            grad: Default::default()
        };

        Value {
            value_inner: Rc::new(RefCell::new(value_inner))
        }
    }
}

impl Add<Value> for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        let value_inner = ValueInner {
            eval: self.eval()+rhs.eval(),
            operation: Operation::Add(self, rhs),
            grad: Default::default()
        };
        Value {
            value_inner: Rc::new(RefCell::new(value_inner))
        }
    }
}

impl Sub<Value> for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Self::Output {
        self+(rhs*(-1.0).into())
    }
}

impl Mul<Value> for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        let value_inner = ValueInner {
            eval: self.eval()*rhs.eval(),
            operation: Operation::Mul(self, rhs),
            grad: Default::default()
        };
        Value {
            value_inner: Rc::new(RefCell::new(value_inner))
        }
    }
}


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
