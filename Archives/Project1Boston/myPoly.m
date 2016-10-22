function Xpoly = myPoly(X,degree)
%MYPOLY is used for polynomial tranformation of a given degree

Xpoly = [];
for d = 1:degree
    Xpoly = [Xpoly, X.^d];

end

end